# spectral_analysis.py written by Mike Lusignan 2012
# revised API by Kyler Brown 2017
from __future__ import division
import numpy as nx

# The following code attempts to grab the FFT
# from scipy, if available.  Otherwise, the FFT from
# numpy is used.
try:
    import scipy.fftpack
    FFT_FN = scipy.fftpack.fft
    USE_SCIPY = True
except:
    FFT_FN = nx.fft.fft
    USE_SCIPY = False
# Load fast C multi-taper library if installed
try:
    import multitaper
    USE_MULTITAPER_LIB = True
except ImportError:
    USE_MULTITAPER_LIB = False


class Spectrum:
    """Computes time-freqeuncy spectral features on continuously sampled signals.
    Spectral features based on Sound Analysis Pro (SAP)."""
    def __init__(self, signal, rate, fft_fn='SAP', freq_range=None, 
                 NFFT=1024, FFT_advance_ms=1.36, data_window_ms=9.27, 
                 n_tapers=4, NW=1.5):
        """
        signal - a sampled data array
        rate   - the sampling rate of the signal
        fft_fn - the fourier transform function. May be 'SAP', 'multitaper' or a custom function.
                    If 'SAP': uses SAP's multitaper estimate. n_tapers and NW are ignored.
                    If 'multitaper': creates a custom multitaper estimate
                    A custom fourier transform function: 
                        should accept the arguments (time_series, sampling, NFFT)
                        and return (spectrum, freqs)
        NFFT   - number of points in Fourier transform
        FFT_advance_ms - Distance between two consecutive short Fourier transforms, in milliseconds
        data_window_ms - amount of data to include in the short Fourier transform,
                              in milliseconds
        n_tapers - Number of tapers to use in a custom multi-taper Fourier transform estimate
        NW    - multi-taper bandwidth parameter for custom multi-taper Fourier transform estimate
                increasing this value reduces side-band ripple
        """
        sample_rate_ms = rate / 1000
        self._NFFT = NFFT
        self._data_in_window = int(data_window_ms * sample_rate_ms)
        self._noverlap = int((data_window_ms - FFT_advance_ms) * sample_rate_ms)
        self._freq_range = freq_range
        if fft_fn == 'SAP':  # use sap defaults
            self._fft_fn = SAP_fft_fn(data_window_ms, FFT_advance_ms, rate)
        elif fft_fn == 'multitaper':
            self._fft_fn = multi_taper_fft_fn(n_tapers, NFFT, NW, 
                                              data_window_ms, FFT_advance_ms, rate)
        else:
            self._fft_fn = fft_fn
        if signal is not None:
            self.analyze_signal(signal, rate)
        
    def analyze_signal(self, signal, rate):
        # pxx dimensions: F x T x k, F: frequencies
        #                            T: times
        #                            k: tapers
        pxx, freqs, times = \
          multi_taper_psd(signal,
                          rate,
                          NFFT=self._NFFT,
                          noverlap=self._noverlap,
                          freq_range=self._freq_range,
                          data_in_window=self._data_in_window,
                          multi_taper_fn=self._fft_fn)
        self._pxx, self._freqs, self._psd_times = (pxx, freqs, times)
        self._power = None

    def power(self, freq_range=None):
        if self._power is None:
            self._power = nx.sum((self._pxx * nx.conj(self._pxx)).real, axis=2)
        if freq_range is None:
            return self._power, self._freqs, self._psd_times
        freq_slice = nx.where(nx.logical_and(self._freqs > freq_range[0],
                                             self._freqs < freq_range[1]))[0]
        pxx = self._power[freq_slice, :]
        freqs = self._freqs[freq_slice, :]
        return pxx, freqs, self._psd_times

    def freq_of_cumulative_power(self,
                                      freq_range=None,
                                      ratio_of_cum_power=0.5):
        power, freqs, times = self.power(freq_range=freq_range)
        total_power = power.sum(0)
        # avoids divide by zero
        total_power = nx.where(total_power != 0.0, total_power, 1.0)
        power = power / total_power
        # calculate cum power
        nx.cumsum(power, axis=0, out=power)
        # each column should contain the cumulative ratio of power at a given
        # frequency
        below_ratio = power < ratio_of_cum_power

        # freq_indicies should range from 0 .. freq.shape[0]
        # freq_inidices should only exceed freq.shape[0] if every value of a
        #   given column is below the power ratio.  This will occur when the
        #   power sums to zero.  When the power sums to zero, the total power
        #   is set to 1.0 to avoid divide-by-zero errors.  In such cases,
        #   every row will fail to exceed the power ratio,
        #   and the corresponding
        #   sum will be equal to freq.shape (and thus higher than the maximum
        #   index of freqs!)
        freq_indices = below_ratio.sum(axis=0) % freqs.shape[0]
        freq_values = freqs[freq_indices]
        return freq_values, times

    def amplitude(self,
                       freq_range=None,
                       noise_threshold=None,
                       noise_ratio=None):
        noise_power = None
        if (noise_threshold and noise_ratio):
            noise_power = self.power(freq_range=(0, noise_threshold))[0]
            noise_power = noise_power.sum(0)
        else:
            noise_threshold, noise_ratio = (None, None)
        pxx, freqs, times = self.power(freq_range=freq_range)
        amp, amp_time = amplitude(pxx, freqs, self._psd_times, noise_power,
                                  noise_ratio)
        return amp, amp_time

    def wiener_entropy(self, freq_range=None):
        pxx, freqs, times = self.power(freq_range)
        return wiener_entropy(pxx, freqs, self._psd_times)

    def spec_derivative(self, freq_range=None, phi=0):
        # phi = 0 calculates a spec derivative across time
        # phi = pi/2 calculates a spec derivative across frequency
        freq_slice = nx.arange(len(self._freqs))
        if freq_range is not None:
            freq_slice = nx.where((self._freqs > freq_range[0]) & 
                                  (self._freqs < freq_range[1]))[0]
        pxx = self._pxx[freq_slice, :, :]
        S = pxx[:, :, :-1] * nx.conj(pxx[:, :, 1:])
        S = nx.real(S * nx.exp(1j * phi))
        temp_sum = nx.sum(S, axis=2)
        #    taper_count = pxx.shape[2] # size of dimension k
        #    temp_sum = nx.zeros(pxx.shape[:-1])
        #    for taper in range(taper_count-1):
        #      temp_sum += nx.real(pxx[:,:,taper]*pxx[:,:,taper+1])
        return temp_sum, self._freqs[freq_slice], self._psd_times

    def time_derivative(self, freq_range=None):
        return self.spec_derivative(freq_range, phi=0)

    def freq_derivative(self, freq_range=None):
        return self.spec_derivative(freq_range, phi=nx.pi / 2)

    def freq_modulation(self, freq_range=None):
        max_time_deriv = nx.max(
            self.time_derivative(freq_range=freq_range)[0]**2,
            axis=0)
        max_freq_deriv = nx.max(
            self.freq_derivative(freq_range=freq_range)[0]**2,
            axis=0)
        fm = nx.arctan(max_time_deriv / max_freq_deriv)
        return fm, self._psd_times

    def max_spec_derivative(self, freq_range=None):
        """
         NOTE: FM is NOT the angle of maximum strength
         (freq)
         |                  .
         |                .
         |              .  ^
         |            .    |
         |          .      \------ FM angle
         |        .
         |      .
         |    .
         |  .   <-------- theta, angle of max strength
         |._____________________
                 (time)
        FM is pi/2 - that angle (recip. of arctan ratio)
        So, instead of
         time_deriv*cos(pi/2 - FM) + freq_deriv*sin(pi/2 - FM)
        it is -time_deriv*sin(FM) + freq_deriv*cos(FM)
        """
        time_deriv = self.time_derivative(freq_range=freq_range)
        freqs = time_deriv[1]
        times = time_deriv[2]
        time_deriv = time_deriv[0]
        freq_deriv = self.freq_derivative(freq_range=freq_range)[0]
        fm = nx.arctan(nx.max(time_deriv**2,
                              axis=0) / nx.max(freq_deriv**2,
                                               axis=0))
        max_spec_deriv = -time_deriv * nx.sin(fm) + freq_deriv * nx.cos(fm)
        return max_spec_deriv, freqs, times

    def goodness_of_pitch(self, freq_range=None):
        max_ds, freqs, times = self.max_spec_derivative(
            freq_range=freq_range)
        pxx, freqs, times = self.power(freq_range)
        norm_max_ds = max_ds / pxx
        estimates2 = FFT_FN(norm_max_ds, axis=0)
        g_of_p = nx.max(nx.abs(estimates2), axis=0)
        return g_of_p, times
    def spectrogram(self, ax=None, freq_range=None, 
                    dB_thresh=35, derivative=True, colormap='inferno'):
        """Plots a spectrogram, requires matplotlib
        ax - axis on which to plot
        freq_range - a tuple of frequencies, eg (300, 8000)
        dB_thresh  - noise floor threshold value, increase to suppress noise, decrease
                        to improve detail
        derivative - if True, plots the spectral derivative, SAP style
        colormap   - colormap to use, good values: 'inferno', 'gray'
        
        Returns an axis object
        """
        from matplotlib import colors
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        if derivative:
            pxx, f, t = self.max_spec_derivative(freq_range=freq_range)
            thresh = value_from_dB(dB_thresh, pxx)
            ax.pcolorfast(t, f, pxx, cmap=colormap, 
                          norm=colors.SymLogNorm(linthresh=thresh))
        else:
            pxx, f, t = self.power(freq_range)
            thresh = value_from_dB(dB_thresh, pxx)
            ax.pcolorfast(t, f, pxx, cmap=colormap,
                          norm=colors.LogNorm(vmin=thresh))
        return ax


def value_from_dB(dB, pxx):
    """Converts a dB value to an absolute value, relative to the maximum value in pxx"""
    return 10 ** (nx.log10(nx.max(nx.abs(pxx))) - dB / 10)


def multi_taper_fft_fn(n_tapers, NFFT=1024, NW=1.5, data_window=9.27, FFT_advance=1.36, sample_rate=44.100e3):
    """Creates a generic multi-taper fft function
    
  Parameters:
    data_window : size of the data window, in ms
    FFT_advance : time interval to advance window for
                  each subsequent FFT
    sample_rate : sample rate, in samples/ms
  Returns:
    an fft_fn
    """
    sample_rate_ms = sample_rate / 1000
    data_in_window = int(data_window * sample_rate_ms)
    noverlap = int(data_in_window - FFT_advance * sample_rate_ms)
    tapers = calc_tapers(data_in_window, NW / data_in_window, n_tapers)
    lambdas = calc_lambdas(data_in_window, NW / data_in_window, tapers)
    padded_tapers = nx.zeros((n_tapers, NFFT))
    padded_tapers[:, :data_in_window] = tapers
    return get_multi_taper_fn(padded_tapers, lambdas, NFFT)

def SAP_fft_fn(data_window=9.27, FFT_advance=1.36, sample_rate=44.100e3):
    """
  Generates a multitaper fft function compatible with the FFT in SAP.
  Ideally, the FFT in scipy will be used (SAP uses FFTW
  for FFTs, as does scipy).

  SAP uses a two tapered MTM.  The two tapers are the
  first two (i.e. two whose lambda values are closest to
  one) in a calculation for 4 tapers.  Tapers are not
  weighted by lambda value.  The normalization of
  tapers is broken, and the tapers are multiplied
  by a scaling factor to replicate the scaling factor
  in SAP.  SAP uses a bandwidth parameter of
  1.5 and a fixed FFT size of 1024 points.

  Parameters:
    data_window : size of the data window, in ms
    FFT_advance : time interval to advance window for
                  each subsequent FFT
    sample_rate : sample rate, in samples/ms
  Returns:
    an fft_fn
  """
    sample_rate_ms = sample_rate / 1000
    NW = 1.5
    NFFT = 1024
    data_in_window = int(data_window * sample_rate_ms)
    noverlap = int(data_in_window - FFT_advance * sample_rate_ms)

    tapers4 = calc_tapers(data_in_window, NW / data_in_window, 4)
    lambdas4 = calc_lambdas(data_in_window, NW / data_in_window, tapers4)
    # take only first two tapers and fill out to use full NFFT
    # length
    tapers = nx.zeros((2, NFFT))
    tapers[:, :data_in_window] = tapers4[0:2, :]
    # no taper weighting done by SAP
    lambdas = nx.ones((2, ))
    # normalization on tapers is broken (see multitap() in tapers.c)
    tapers = tapers * nx.sqrt(data_in_window)
    return get_multi_taper_fn(tapers, lambdas, NFFT)

# --- spectral feature functions ---


def noise_filter(psd, freqs, amp, t, noise_ratio=0.5, noise_cutoff=500):
    noise_index = nx.max(nx.where(freqs < noise_cutoff)[0])
    noise_power = psd[0:noise_index, :].sum(0)
    return noise_power / amp < noise_ratio


def amplitude(psd, freqs, t, noise_power=None, noise_ratio=None, baseline=70):
    """
  Calculates the amplitude in Db of the time series.

  Parameters:
    psd:    PSD, an M x N matrix of power at frequency (M) and time (N)
    freqs:  length M vector of frequencies in psd
    t:      length N vector of time values
    baseline: baseline for calculating Db
  Returns:
    amp : N length vector of amplitude values
    t   : N length vector of times
  """
    amp = psd.sum(0)
    if noise_power is not None:
        small_amp = nx.where(amp < 1)[0]
        amp[small_amp] = 1
        noise_filter = noise_power / amp < noise_ratio
        amp = amp * noise_filter
    amp = nx.log10(amp + 1) * 10 - baseline
    return amp, t


def power_limited_wiener_entropy(psd, freqs, t, power_limit=0.9):
    """Calculates the Wiener entropy for the PSD, using frequency limits
     at each time point determined by the percentage of the cumulative
     power at each time point.

     Wiener entropy for a power spectrum is defined as the log of geometric
     mean divided by the arithmetic mean.

     psd:         PSD, an M x N matrix of power at frequency (M) and time (N)
     freqs:       length M vector of frequencies in psd
     t:           length N vector of time values
     power_limit: percentage of cumulative power to include (default 90%)
                  supplied as decimal fraction (i.e. 100% => 1.0)

     returns
       w: N length vector of Wiener entropies
       t: N length vector of times
  """
    cum_power = psd.cumsum(axis=0)
    cum_power = cum_power / psd.sum(axis=0)
    within_power_limit = cum_power < power_limit
    limited_psd = psd * within_power_limit
    freqs = within_power_limit.sum(axis=0)
    arth_mean = limited_psd.sum(axis=0) / freqs
    log_psd = limited_psd + (cum_power > power_limit)
    log_psd = nx.log(log_psd)
    log_geo_mean = log_psd.sum(axis=0) / freqs
    wiener = log_geo_mean - nx.log(arth_mean)
    return wiener, t


def wiener_entropy(psd, freqs, t):
    """Calculates the Wiener entropy for the PSD.

     Wiener entropy for a power spectrum is defined as the log of geometric
     mean divided by the arithmetic mean.

     psd:   PSD, an M x N matrix of power at frequency (M) and time (N)
     freqs: length M vector of frequencies in psd
     t:     length N vector of time values

     returns
       w: N length vector of Wiener entropies
       t: N length vector of times
  """
    freqs = psd.shape[0]
    arth_mean = psd.sum(0) / freqs
    pxx_t_log = nx.log(psd)
    log_geo_mean = nx.sum(pxx_t_log, 0) / freqs
    wiener = log_geo_mean - nx.log(arth_mean)
    return wiener, t

def spec_deriv_reduce_fn(estimates):
    # estimates2 is K x N array, where K is taper number
    # and N is frequency number
    #
    # To estimate spectral derivative:
    # Re{ exp(j*phi) * 1/(K-1) * sum_over_k(taper_k * conj(taper_k+1)) }
    #   where phi is the direction of the derivative
    x = estimates
    K = x.shape[0]
    x_conj = x[1:, :].conj()
    x = x[:-1, :]
    sum_over_k = nx.sum(x * x_conj, axis=0) / (K - 1)
    time_deriv = sum_over_k.real
    freq_deriv = sum_over_k.imag
    # which angle theta to use?
    # choose theta corresponding to maximum magnitude
    # maximum magnitude at each point is arctan(freq_deriv/time_deriv)

    # as sap does it....
    #  max_time_deriv = nx.max(time_deriv**2)
    #  max_freq_deriv = nx.max(freq_deriv**2)
    #  angle = nx.arctan(max_time_deriv/max_freq_deriv)
    #  return -time_deriv*nx.sin(angle) + freq_deriv*nx.cos(angle)

    angles = nx.arctan(-time_deriv / freq_deriv)
    sine_angles = nx.sin(angles)
    cosine_angles = nx.cos(angles)
    angle_index = nx.argmax(nx.abs(cosine_angles * time_deriv - sine_angles *
                                   freq_deriv))
    max_angle = angles[angle_index]
    max_angle1 = max_angle
    max_angle2 = (max_angle + nx.pi) % (2 * nx.pi)
    max_derivs = nx.array((time_deriv[angle_index], -freq_deriv[angle_index]))
    max_value1 = nx.sum(max_derivs *
                        nx.array((nx.cos(max_angle1), nx.sin(max_angle1))))
    max_value2 = nx.sum(max_derivs *
                        nx.array((nx.cos(max_angle2), nx.sin(max_angle2))))
    if max_value2 > max_value1:
        max_angle = max_angle2
    max_cos, max_sin = (nx.cos(max_angle), nx.sin(max_angle))
    spec_deriv = time_deriv * max_cos - freq_deriv * max_sin
    return spec_deriv

def multi_taper_psd(signal,
                    rate,
                    NFFT,
                    noverlap,
                    freq_range,
                    data_in_window,
                    multi_taper_fn):
    """
  Calculates an MTM PSD from the signal.

  Parameters:
    signal         : vector of data
    rate           : sampling rate, in samples/sec
    tapers         : NxM matrix of tapers, where each column
                     is a taper of length N
    lambdas        : vector of M lambdas for M tapers
    NFFT           : size of FFT window
    noverlap       : window overlap, in points
    freq_range     : range of frequencies to include in Hz,
                     as tuple
    data_in_window : if specified, subset of NFFT point count
                     to include in FFT calcuation
  Returns:
    pxx   : NxM matrix of power values at each frequency
    freqs : vector of size N containing frequency at each index
            N
    times : vector of size M containing times corresponding to
            each index M
  """
    # NFFT determines the number of windows
    if  data_in_window > NFFT:
        data_in_window = NFFT
        print('warning, data window larger than NFFT')
    window_count = int(len(signal) / (data_in_window - noverlap))
    if (window_count * (data_in_window - noverlap) < len(signal)):
        window_count += 1
        expanded_signal = nx.zeros((window_count * (data_in_window - noverlap),
                                    ))
        expanded_signal[0:len(signal)] = signal
        signal = expanded_signal
    window_starts = nx.arange(window_count) * (data_in_window - noverlap)
    rate = rate
    times = window_starts / rate
    pxx = None
    window_index = 0
    for window_start in window_starts:
        signal_interval = signal[window_start:(window_start + data_in_window)]
        spectrum, freqs = multi_taper_fn(signal_interval, rate, NFFT)
        taper_count = spectrum.shape[1]
        if pxx is None:
            pxx = nx.zeros(
                (len(freqs), len(window_starts), taper_count),
                dtype=complex)
        pxx[:, window_index, :] = spectrum
        window_index += 1

    if freq_range is not None:
        d_freq = freqs[1] - freqs[0]
        min_freq, max_freq = freq_range

        min_index, max_index = (min_freq / d_freq, max_freq / d_freq)
        freqs = freqs[min_index:max_index]
        pxx = pxx[min_index:max_index, :]

    return pxx, freqs, times

# --- Multi-taper machinery ---

def taper_diags(N, W):
    t = nx.arange(0, N)
    diag = (N - 1. - 2. * t)**2. / 4. * nx.cos(2 * nx.pi * W)
    t = nx.arange(1, N)
    off_diag = t * (N - t) / 2.0
    return diag, off_diag


def taper_matrix(N, W):
    """
    Generates the matrix used for taper calculations.
    """
    N = int(N)
    m = nx.zeros((N, N))
    n = 0
    # diagonal
    diag, off_diag = taper_diags(N, W)
    diag_indices = nx.arange(0, N) * (N + 1)
    nx.put(m, diag_indices, diag)
    nx.put(m, (diag_indices[0:-1] + 1), off_diag)
    nx.put(m, (diag_indices[1:] - 1), off_diag)
    return m


def calc_tapers(N, W, taper_count):
    """
  Generates a matrix of tapers appropriate for a multi-tapered
  analysis.  Tapers are eigenvectors of the discrete prolean
  sequence.
    N: taper size
    W: bandwidth (often 1.5/N)
    taper_count: number of desired tapers
  Returns taper_count x N matrix of tapers.
  """
    tapers = nx.zeros((N, taper_count))
    m = nx.zeros((N, N))
    n = 0
    m = taper_matrix(N, W)
    vals, vects = nx.linalg.eigh(m)
    # find largest eigenvalues
    taper_indices = []
    vals_sorted = vals.copy()
    vals_sorted.sort()
    # make sure tapers are examined from largest
    # to smallest wrt eigenvalue
    biggest_vals = list(vals_sorted[-taper_count:])
    biggest_vals.reverse()
    for t in range(taper_count):
        index = nx.where(vals == biggest_vals[t])[0]
        index = index[0]
        taper_indices.append(index)
    # note - tapers are normalized
    # apply Slepian proscribes polarity constrants
    for k in range(len(taper_indices)):
        taper = vects[:, taper_indices[k]]
        if k % 2 == 0:
            # even
            if nx.sum(taper) < 0:
                taper = -taper
        else:
            if nx.sum((N - 1 - nx.arange(N)) * taper) < 0:
                taper = -taper
        tapers[:, k] = taper
    return tapers.transpose()


def sinc_matrix(N, W):
    A = nx.zeros((N, N))
    t = nx.arange(1, N)
    master_row = nx.sin(2 * nx.pi * W * t) / (nx.pi * t)
    master_row = list(master_row[::-1]) + [2 * W] + list(master_row)
    master_row = nx.array(master_row)
    row = nx.arange(0, N) + N - 1
    for r in nx.arange(N):
        A[r, :] = master_row[row - r]
    return A


def calc_lambdas(N, W, tapers):
    A = sinc_matrix(N, W)
    taper_count = tapers.shape[0]
    lambdas = []
    for t in nx.arange(taper_count):
        taper = tapers[t, :]
        LHS = nx.dot(A, taper)
        RHS = taper
        lam = nx.mean(LHS / RHS)
        lambdas.append(lam)
    return nx.array(lambdas)


def get_multi_taper_fn(tapers, lambdas, NFFT):
    if USE_MULTITAPER_LIB:
        # Returns a fast multi-taper function, requires the multitaper library
        mt = multitaper.multi_taper_estimate(tapers, lambdas)
        def mt_fn(time_series, sampling, NFFT):
            pxx = mt(time_series)
            freqs = nx.arange(len(pxx)) / len(pxx) * sampling / 2
            return pxx, freqs
        return mt_fn
    else:
        # Use a pure python multi-taper implemenation
        def mt_fn(time_series, sampling, NFFT):
            return multi_taper(time_series,
                               sampling,
                               tapers=tapers,
                               lambdas=lambdas,
                               NFFT=NFFT)
        return mt_fn

def multi_taper(signal, rate, tapers, lambdas, NFFT):
    """ Calculates a single MTM.

  Parameters:
    signal      : raw data
    rate        : sampling rate, in samples/sec
    tapers      : NxM matrix of M tapers of length N
                : (tapers are columns of matrix)
    lambdas     : vector of M lambda weights of M tapers
    NFFT        : size of transform
  Return:
    spec  : vector of length N containing power spectrum
    freqs : vector of length N containing frequencies
            corresponding to each N index
  """
    N = len(signal)
    if N != NFFT:
        if N < NFFT:
            s = nx.zeros((NFFT, ))
            s[0:int(N)] = signal
            signal = s
        else:
            signal = signal[0:NFFT]
        N = NFFT

    # generate weighted signals
    weights = lambdas
    # generates an taper_count x N array, where
    # each row (k) is signal * taper (k)
    estimates2 = signal * tapers
    # calculates 1D FFT for each row in one call
    estimates2 = FFT_FN(estimates2, axis=1)
    estimates2 = estimates2[:, 0:(N // 2 + 1)] * weights[:, nx.newaxis]
    spectrum2 = estimates2.transpose()

    ### I used the code below to verify that
    ### the optimized code above calculates expected
    ### values.

    #  estimates = nx.zeros((taper_count, N/2 + 1))
    #  for t in range(taper_count):
    #    weight = 1/N * lambdas[t]
    #    estimate = nx.fft.fft(signal*tapers[t,:])
    #    estimate = estimate[0:(N/2 + 1)]
    #    estimate = nx.square(nx.abs(estimate))
    #    estimates[t, :] = estimate * weight
    ##  spectrum = estimates.sum(0) ** 0.5
    #  spectrum = estimates.sum(0)
    #  if (nx.any(spectrum != spectrum2)):
    #    print 'Not equal!'

    max_freq = rate / 2
    freqs = nx.arange(0, max_freq, max_freq / (N / 2 + 1))
    return spectrum2, freqs

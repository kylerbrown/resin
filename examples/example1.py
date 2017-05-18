import matplotlib.pyplot as plt
from scipy.io import wavfile
import resin
import seaborn
seaborn.set_style('white')

# loading in the data
sr, data = wavfile.read('black33.wav')
# create a Spectra object using SAP-like defaults
spa = resin.sap_spectra(sr)
spa.signal(data)
spa.spectrogram()
plt.savefig('example1_sap.png')
plt.cla()

# Alternately, customize the mutli-taper parameters
# by creating a Spectra object directly.

spa2 = resin.Spectra(sr, 
                    NFFT=1024, 
                    noverlap=1000, 
                    data_window=int(0.01 * sr), 
                    n_tapers=3, 
                    NW=1.8,
                    freq_range=(300, 9000))
spa2.signal(data)
spa2.spectrogram()
plt.savefig('example1_custom_spectrogram.png')

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def read_data(index):

    f = open('Data Sets/signal' + str(index) + '.dat', 'r')

    lines = f.read().split('\n')
    l = lines.__len__() - 1
    time = np.zeros(l)
    wave_data = np.zeros(l)
    noise_data = np.zeros(l)

    for i in range(0, l):
        time[i] = float(lines[i].split(' ')[0])
        wave_data[i] = float(lines[i].split(' ')[1])

    f.close()

    f = open('Data Sets/noise' + str(index) + '.dat', 'r')
    lines = f.read().split('\n')
    for i in range(0, l):
        noise_data[i] = float(lines[i].split(' ')[1])

    f.close()

    return time, wave_data, noise_data


time, wave_data, noise_data = read_data(1)


sampF = 1/(time[1]-time[0])

plt.figure(1)
plt.plot(time, wave_data, time, noise_data)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.grid(True,'both')
plt.draw()

f, t, Sxx = signal.spectrogram(wave_data, sampF, 'hann', 100, 90)

plt.figure(2)
plt.pcolormesh(t, f, np.log10(Sxx))
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.yscale('log')
plt.ylim([10, 10**3])
plt.draw()

f, t, Sxx = signal.spectrogram(noise_data, sampF, 'hann', 100, 90)

plt.figure(3)
plt.pcolormesh(t, f, np.log10(Sxx))
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.yscale('log')
plt.ylim([10, 10**3])
plt.draw()

plt.figure(4)
f, Pxx_den = signal.welch(wave_data, sampF)
plt.loglog(f, Pxx_den)
f, Pxx_den = signal.welch(noise_data, sampF)
plt.loglog(f, Pxx_den)
# plt.xlim([10, 10**3])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [strain^2/Hz]')
plt.grid(True,'both')
plt.draw()

plt.show()
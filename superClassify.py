import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def read_data(index):

    f = open('Data Sets/signal' + str(index) + '.dat', 'r')

    lines = f.read().split('\n')
    l = lines.__len__() - 1
    time = np.zeros(l)
    wave_data = np.zeros(l)
    noise_data = np.zeros(l)

    for i in range(0, l):
        time[i] = float(lines[i].split(' ')[0])
        wave_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    f = open('Data Sets/noise' + str(index) + '.dat', 'r')
    lines = f.read().split('\n')
    for i in range(0, l):
        noise_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    return time, wave_data, noise_data

x = np.zeros(4096)
y = np.array(0)
for i in range(1, len(os.listdir('./Data Sets/'))):

    if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
        time, wave_data, noise_data = read_data(i)

    x = np.column_stack((x, wave_data))
    y=np.append(y, 1)
    x = np.column_stack((x, noise_data))
    y = np.append(y, 0)


print(x.shape)
print(y.shape)
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


for i in range(1, len(os.listdir('./Data Sets/'))):

    if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
        time, wave_data, noise_data = read_data(i)

    print(len(time))
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


x_train = np.zeros(4096)
y_train = np.array(0)
x_test = np.zeros(4096)
y_test = np.array(0)

for i in range(1, len(os.listdir('./Data Sets/'))):
    if i<=len(os.listdir('./Data Sets/'))/2:
        if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
            time, wave_data, noise_data = read_data(i)

        x_train = np.column_stack((x_train, wave_data))
        y_train = np.append(y_train, 1)
        x_train = np.column_stack((x_train, noise_data))
        y_train = np.append(y_train, 0)

    if i > len(os.listdir('./Data Sets/')) / 2:
        if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
            time, wave_data, noise_data = read_data(i)

        x_test = np.column_stack((x_test, wave_data))
        y_test = np.append(y_test, 1)
        x_test = np.column_stack((x_test, noise_data))
        y_test = np.append(y_test, 0)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
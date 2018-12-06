import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
import os
import h5py
import itertools
import time

def read_data(index, gain):

    f = open('CBC Data/Gain'+str(gain)+'/signal' + str(index) + '.dat', 'r')

    lines = f.read().split('\n')
    l = lines.__len__() - 1
    tim = np.zeros(l)
    wave_data = np.zeros(l)
    noise_data = np.zeros(l)

    for i in range(0, l):

        if not (np.isnan(float(lines[i].split(' ')[1]))):
            tim[i] = float(lines[i].split(' ')[0])
            wave_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    f = open('CBC Data/Gain'+str(gain)+'/noise' + str(index) + '.dat', 'r')
    lines = f.read().split('\n')
    l = lines.__len__() - 1
    for i in range(0, l):
        if not(np.isnan(float(lines[i].split(' ')[1]))):
            noise_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    return tim, wave_data, noise_data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 32)
        self.pool = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(16128, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = x.expand(1, 1, 2049)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16128)
        x = self.fc1(x)
        x = self.out(x)
        return x

gain = 0.1

x_train = np.zeros(2049)
y_train = np.array(0)
x_test = np.zeros(2049)
y_test = np.array(0)

for i in range(1, len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/2+1):
    if i <= len(os.listdir('./CBC Data/Gain' + str(gain) + '/')) / 4:
        # if os.path.isfile('CBC Data/Gain' + str(gain) + '/signal' + str(i) + '.dat') & \
        #         os.path.isfile('CBC Data/Gain' + str(gain) + '/noise' + str(i) + '.dat'):

        tim, wave_data, noise_data = read_data(i, gain)

        f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096)
        f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096)

        with np.errstate(divide='raise'):

            x_train = np.column_stack((x_train, np.log10(P2.T)))
            y_train = np.append(y_train, 1)
            x_train = np.column_stack((x_train, np.log10(P1.T)))
            y_train = np.append(y_train, 0)
            #
            # x_train = np.column_stack((x_train, wave_data))
            # y_train = np.append(y_train, 1)
            # x_train = np.column_stack((x_train, noise_data))
            # y_train = np.append(y_train, 0)

    if i > len(os.listdir('./CBC Data/Gain' + str(gain) + '/')) / 4:
        # if os.path.isfile('CBC Data/Gain' + str(gain) + '/signal' + str(i) + '.dat') & \
        #         os.path.isfile('CBC Data/Gain' + str(gain) + '/noise' + str(i) + '.dat'):

        tim, wave_data, noise_data = read_data(i, gain)

        f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096)
        f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096)

        with np.errstate(divide='raise'):

            x_test = np.column_stack((x_test, np.log10(P2.T)))
            y_test = np.append(y_test, 1)
            x_test = np.column_stack((x_test, np.log10(P1.T)))
            y_test = np.append(y_test, 0)


train_data = torch.from_numpy((x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1))/np.std(x_train[:, 1:])).float()
test_data = torch.from_numpy((x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1))/np.std(x_test[:, 1:])).float()
train_labels = torch.from_numpy(y_train[1:]).float()
test_labels = torch.from_numpy(y_test[1:]).float()

net = Net()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=10**-5, momentum=0.1)

epochLim = 10

testAcc = np.zeros(epochLim)
trainAcc = np.zeros(epochLim)
for epoch in range(epochLim):

    running_loss = 0.0
    for i in range(0, len(train_data)):

        optimizer.zero_grad()

        output = net(train_data[i])
        loss = criterion(output.view(-1), train_labels[i].view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    correct = 0.0
    with torch.no_grad():
        for j in range(len(test_data)):
            output = net(test_data[j])
            predicted = round(float(output.data))
            correct += (predicted == test_labels[j]).item()
    print('Test accuracy: %d %%' % (
            100 * correct / test_labels.size(0)))
    testAcc[epoch] = 100 * correct / test_labels.size(0)

    correct = 0.0
    with torch.no_grad():
        for j in range(len(train_data)):
            output = net(train_data[j])
            predicted = round(float(output.data))
            correct += (predicted == train_labels[j]).item()
    print('Train accuracy: %d %%' % (
            100 * correct / train_labels.size(0)))
    trainAcc[epoch] = 100 * correct / train_labels.size(0)

print('Finished Training')


gainList = np.array((0.001, 0.002, 0.003, 0.004, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
					 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0))
#, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,

gainAcc = np.zeros(gainList.size)
gainIndex = 0

for gain in gainList:

    x_test = np.zeros(2049)
    y_test = np.array(0)

    for i in range(1, len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/2+1):

            tim, wave_data, noise_data = read_data(i, gain)

            f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096)
            f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096)

            with np.errstate(divide='raise'):

                x_test = np.column_stack((x_test, np.log10(P2.T)))
                y_test = np.append(y_test, 1)
                x_test = np.column_stack((x_test, np.log10(P1.T)))
                y_test = np.append(y_test, 0)


    test_data = torch.from_numpy(
        (x_test[:, 1:].T - np.mean(x_test[:, 1:], axis=1)) / np.std(x_test[:, 1:])).float()
    test_labels = torch.from_numpy(y_test[1:]).float()

    correct = 0.0
    with torch.no_grad():
        for j in range(len(test_data)):
            output = net(test_data[j])
            predicted = round(float(output.data))
            correct += (predicted == test_labels[j]).item()

    print('Accuracy on '+str(1/gain)+' Mpc dataset: %d %%' % (
            100 * correct / test_labels.size(0)))
    gainAcc[gainIndex] = 100 * correct / test_labels.size(0)

    gainIndex += 1

for filename in os.listdir('GW Events'):

    f = h5py.File('GW Events/'+filename, 'r')
    data = np.array(f['strain']['Strain'])
    f.close()
    predicted = 0.0
    for i in range(0, len(data), 4096):
        f, P = signal.welch(data[i:i+4096], fs=4096, nperseg=4096)

        x = torch.from_numpy((np.log10(P) - np.mean(np.log10(P))) / np.std(np.log10(P))).float()

        with torch.no_grad():
            output = net(x)
            predicted += round(float(output.data))

    print(predicted)

plt.figure()
plt.plot(gainList, gainAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Gain')
plt.draw()
plt.savefig('NNAccuracyCBC.pdf')

plt.figure()
plt.plot(1/gainList, gainAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Distance (Mpc)')
plt.draw()
plt.savefig('NNAccuracyDistanceCBC.pdf')

plt.figure()
plt.plot(range(epochLim), trainAcc)
plt.plot(range(epochLim), testAcc)
plt.legend(('Training', 'Testing'))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.draw()
plt.savefig('NNTraining.pdf')
plt.show()
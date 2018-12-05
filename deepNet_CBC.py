import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
import os
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
        self.conv1 = nn.Conv1d(1, 20, 250)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(38460, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = x.expand(1, 1, 4096)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 38460)
        x = self.fc1(x)
        x = self.out(x)
        return x


gainList = np.array((0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10))


gainIndex = 0

gain = 0.5

x_train = np.zeros(4096)
y_train = np.array(0)
x_test = np.zeros(4096)
y_test = np.array(0)

for i in range(1, len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/10):
    if i<=len(os.listdir('./CBC Data/Gain'+str(gain)+'/'))/2:
        if os.path.isfile('CBC Data/Gain'+str(gain)+'/signal' + str(i) + '.dat') & \
            os.path.isfile('CBC Data/Gain'+str(gain)+'/noise' + str(i) + '.dat'):
                tim, wave_data, noise_data = read_data(i, gain)

    with np.errstate(divide='raise'):
        try:
            x_train = np.column_stack((x_train, wave_data))
            y_train = np.append(y_train, 1)
            x_train = np.column_stack((x_train, noise_data))
            y_train = np.append(y_train, 0)
        except FloatingPointError:
            print('Error skipping this data point')

    if i > len(os.listdir('./CBC Data/Gain'+str(gain)+'/')) / 2:
        if os.path.isfile('CBC Data/Gain'+str(gain)+'/signal' + str(i) + '.dat') & \
            os.path.isfile('CBC Data/Gain'+str(gain)+'/noise' + str(i) + '.dat'):
                tim, wave_data, noise_data = read_data(i, gain)

    with np.errstate(divide='raise'):
        try:
            x_test = np.column_stack((x_test, wave_data))
            y_test = np.append(y_test, 1)
            x_test = np.column_stack((x_test, noise_data))
            y_test = np.append(y_test, 0)
        except FloatingPointError:
            print('Error skipping this data point')


train_data = torch.from_numpy((x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1))/np.std(x_train[:, 1:])).float()
test_data = torch.from_numpy((x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1))/np.std(x_test[:, 1:])).float()
train_labels = torch.from_numpy(y_train[1:]).float()
test_labels = torch.from_numpy(y_test[1:]).float()

net = Net()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=2*10**-3, momentum=0.1)

epochLim = 100

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

    correct = 0
    total = 0
    with torch.no_grad():
        for j in range(len(test_data)):
            output = net(test_data[j])
            predicted = round(output.data)
            correct += (predicted == test_labels[j]).item()

    print('Test accuracy: %d %%' % (
            100 * correct / test_labels.size(0)))
    testAcc[epoch] = 100 * correct / test_labels.size(0)

    correct = 0
    total = 0
    with torch.no_grad():
        for j in range(len(train_data)):
            outputs = net(train_data[j])
            predicted = round(outputs.data)
            correct += (predicted == train_labels[j]).item()

    print('Train accuracy: %d %%' % (
            100 * correct / train_labels.size(0)))

    trainAcc[epoch] = 100 * correct / train_labels.size(0)

print('Finished Training')

plt.figure()
plt.plot(range(epochLim), trainAcc)
plt.plot(range(epochLim), testAcc)
plt.legend(('Training', 'Testing'))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
# plt.savefig('3_convNet.pdf')
plt.show()
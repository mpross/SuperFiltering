import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# Data reading
def read_data(index, gain):

    f = open('Supernova Data/Gain'+str(gain)+'/signal' + str(index) + '.dat', 'r')

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

    f = open('Supernova Data/Gain'+str(gain)+'/noise' + str(index) + '.dat', 'r')
    lines = f.read().split('\n')
    l = lines.__len__() - 1
    for i in range(0, l):
        if not(np.isnan(float(lines[i].split(' ')[1]))):
            noise_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    return tim, wave_data, noise_data


# Convolutional neural net definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 1000, 100)
        self.pool = nn.MaxPool1d(1)
        self.fc1 = nn.Linear(30000, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = x.expand(1, 1, 129)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 30000)
        x = self.fc1(x)
        x = self.out(x)
        return x


# Gain to train the CNN on
gain = 0.3

x_train = np.zeros(129)
y_train = np.array(0)
x_test = np.zeros(129)
y_test = np.array(0)

# Splitting data into test and training sets
for i in range(1, len(os.listdir('./Supernova Data/Gain' + str(gain) + '/'))/2+1):
    if i <= len(os.listdir('./Supernova Data/Gain' + str(gain) + '/')) / 4:
        if os.path.isfile('Supernova Data/Gain' + str(gain) + '/signal' + str(i) + '.dat') & \
                os.path.isfile('Supernova Data/Gain' + str(gain) + '/noise' + str(i) + '.dat'):
            tim, wave_data, noise_data = read_data(i, gain)

        # Power spectra
        f, P1 = signal.welch(noise_data, fs=4096)
        f, P2 = signal.welch(wave_data, fs=4096)

        with np.errstate(divide='raise'):
            try:
                # Data stacking, 1 GW, 0 noise
                x_train = np.column_stack((x_train, np.log10(P2)))
                y_train = np.append(y_train, 1)
                x_train = np.column_stack((x_train, np.log10(P1)))
                y_train = np.append(y_train, 0)
            except FloatingPointError:
                print('Error skipping this data point')

    if i > len(os.listdir('./Supernova Data/Gain' + str(gain) + '/')) / 4:
        if os.path.isfile('Supernova Data/Gain' + str(gain) + '/signal' + str(i) + '.dat') & \
                os.path.isfile('Supernova Data/Gain' + str(gain) + '/noise' + str(i) + '.dat'):
            tim, wave_data, noise_data = read_data(i, gain)

        # Power spectra
        f, P1 = signal.welch(noise_data, fs=4096)
        f, P2 = signal.welch(wave_data, fs=4096)

        with np.errstate(divide='raise'):
            try:
                # Data stacking, 1 GW, 0 noise
                x_test = np.column_stack((x_test, np.log10(P2)))
                y_test = np.append(y_test, 1)
                x_test = np.column_stack((x_test, np.log10(P1)))
                y_test = np.append(y_test, 0)
            except FloatingPointError:
                print('Error skipping this data point')

# Cut off first zero, normalize, and turn into tensor
train_data = torch.from_numpy((x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1))/np.std(x_train[:, 1:])).float()
test_data = torch.from_numpy((x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1))/np.std(x_test[:, 1:])).float()
train_labels = torch.from_numpy(y_train[1:]).float()
test_labels = torch.from_numpy(y_test[1:]).float()

# Net initialization, loss and optimizer definition
net = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=5*10**-3, momentum=0.5)

# Net training
epochLim = 200

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
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

    correct = 0.0
    with torch.no_grad():
        for j in range(len(test_data)):
            output = net(test_data[j])
            predicted = round(float(output.data))
            correct += (predicted == test_labels[j]).item()

    print('Test accuracy: %d %%' % (
            100 * correct / test_labels.size(0)))
    testAcc[epoch] = correct / test_labels.size(0)

    correct = 0.0
    with torch.no_grad():
        for j in range(len(train_data)):
            outputs = net(train_data[j])
            predicted = round(outputs.data)
            correct += (predicted == train_labels[j]).item()

    print('Train accuracy: %d %%' % (
            100 * correct / train_labels.size(0)))

    trainAcc[epoch] = correct / train_labels.size(0)

print('Finished Training')

# Apply trained net to other data sets with different gains
gainList = np.array((0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0))
gainAcc = np.zeros(gainList.size)
gainIndex = 0

for gain in gainList:

    x_test = np.zeros(129)
    y_test = np.array(0)

    for i in range(1, len(os.listdir('./Supernova Data/Gain'+str(gain)+'/'))/2+1):

            tim, wave_data, noise_data = read_data(i, gain)

            # Power spectra
            f, P1 = signal.welch(noise_data, fs=4096)
            f, P2 = signal.welch(wave_data, fs=4096)

            with np.errstate(divide='raise'):
                # Data stacking, 1 GW, 0 noise
                x_test = np.column_stack((x_test, np.log10(P2.T)))
                y_test = np.append(y_test, 1)
                x_test = np.column_stack((x_test, np.log10(P1.T)))
                y_test = np.append(y_test, 0)

    # Normalize and convert to tensor
    test_data = torch.from_numpy(x_test[:, 1:].T - np.mean(x_test[:, 1:], axis=1) / np.std(x_test[:, 1:])).float()
    test_labels = torch.from_numpy(y_test[1:]).float()

    correct = 0.0
    with torch.no_grad():
        for j in range(len(test_data)):
            output = net(test_data[j])
            predicted = round(float(output.data))
            correct += (predicted == test_labels[j]).item()

    print('Accuracy on '+str(1/gain)+' Mpc dataset: %d %%' % (
            100 * correct / test_labels.size(0)))
    gainAcc[gainIndex] = correct / test_labels.size(0)

    gainIndex += 1

# Apply a selection of simple methods from sklearn to the data to compare
NNAcc = np.zeros(gainList.size)
NearNAcc = np.zeros(gainList.size)
logRegAcc = np.zeros(gainList.size)
SVMAcc = np.zeros(gainList.size)

gainIndex = 0

for gain in gainList:
    print(str(gain))
    x_train = np.zeros(129)
    y_train = np.array(0)
    x_test = np.zeros(129)
    y_test = np.array(0)

    for i in range(1, len(os.listdir('./Supernova Data/Gain'+str(gain)+'/'))/2+1):
        if i<=len(os.listdir('./Supernova Data/Gain'+str(gain)+'/'))/4:
            if os.path.isfile('Supernova Data/Gain'+str(gain)+'/signal' + str(i) + '.dat') & \
                    os.path.isfile('Supernova Data/Gain'+str(gain)+'/noise' + str(i) + '.dat'):

                    tim, wave_data, noise_data = read_data(i, gain)

            # Power spectra
            f, P1 = signal.welch(noise_data, fs=4096)
            f, P2 = signal.welch(wave_data, fs=4096)

            with np.errstate(divide='raise'):
                try:
                    # Data stacking, 1 GW, 0 noise
                    x_train = np.column_stack((x_train, np.log10(P2)))
                    y_train = np.append(y_train, 1)
                    x_train = np.column_stack((x_train, np.log10(P1)))
                    y_train = np.append(y_train, -1)
                except FloatingPointError:
                    print('Error skipping this data point')

        if i > len(os.listdir('./Supernova Data/Gain'+str(gain)+'/')) / 4:
            if os.path.isfile('Supernova Data/Gain'+str(gain)+'/signal' + str(i) + '.dat') & \
                    os.path.isfile('Supernova Data/Gain'+str(gain)+'/noise' + str(i) + '.dat'):

                    tim, wave_data, noise_data = read_data(i, gain)

            # Power spectra
            f, P1 = signal.welch(noise_data, fs=4096)
            f, P2 = signal.welch(wave_data, fs=4096)

            with np.errstate(divide='raise'):
                try:
                    # Data stacking, 1 GW, 0 noise
                    x_test = np.column_stack((x_test, np.log10(P2)))
                    y_test = np.append(y_test, 1)
                    x_test = np.column_stack((x_test, np.log10(P1)))
                    y_test = np.append(y_test, -1)
                except FloatingPointError:
                    print('Error skipping this data point')

    # Cut off first zero, normalize, and turn into tensor
    x_train = (x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1))/np.std(x_train)
    x_test = (x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1))/np.std(x_test)
    y_train = y_train[1:]
    y_test = y_test[1:]

    # Train and test algorithms
    logReg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000)
    logReg.fit(x_train, y_train.ravel())
    logRegAcc[gainIndex] = logReg.score(x_test, y_test)

    svmAlg = svm.SVC(gamma='scale')
    svmAlg.fit(x_train, y_train.ravel())
    SVMAcc[gainIndex] = svmAlg.score(x_test, y_test)

    NearN = KNeighborsClassifier(10)
    NearN.fit(x_train, y_train.ravel())
    NearNAcc[gainIndex] = NearN.score(x_test, y_test)

    NN = MLPClassifier(max_iter=10000)
    NN.fit(x_train, y_train.ravel())
    NNAcc[gainIndex] = NN.score(x_test, y_test)

    gainIndex += 1

plt.figure()
plt.plot(gainList, logRegAcc)
plt.plot(gainList, SVMAcc)
plt.plot(gainList, NearNAcc)
plt.plot(gainList, NNAcc)
plt.plot(gainList, gainAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Gain')
plt.legend(('Logistic Regression', 'SVM', 'Nearest Neighbor', 'Neural Network', 'Convolutional Neural Network'))
plt.grid(True)
plt.draw()
plt.savefig('SimpleAccuracySN.pdf')

plt.figure()
plt.plot(10/gainList, logRegAcc)
plt.plot(10/gainList, SVMAcc)
plt.plot(10/gainList, NearNAcc)
plt.plot(10/gainList, NNAcc)
plt.plot(10/gainList, gainAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Distance (kpc)')
plt.legend(('Logistic Regression', 'SVM', 'Nearest Neighbor', 'Neural Network', 'Convolutional Neural Network'))
plt.grid(True)
plt.draw()
plt.savefig('SimpleAccuracyDistanceSN.pdf')

plt.figure()
plt.plot(range(epochLim), trainAcc)
plt.plot(range(epochLim), testAcc)
plt.legend(('Training', 'Testing'))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.draw()
plt.savefig('NNTrainingSN.pdf')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import itertools
from sklearn.linear_model import LogisticRegression

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


def logReg(x, y, lamb):

    w = np.zeros(x.shape[0])

    l = sum(np.log(1 + np.exp(-y * np.dot(x.T, w)))) + lamb * np.dot(w.T, w)
    print(l)


    stepSize = 10**-8
    N=10**5
    for i in range(N):
        deltaL = np.dot(-y*np.exp(-y * np.dot(x.T, w))/(1 + np.exp(-y * np.dot(x.T, w))), x.T) \
                 + 2*lamb*np.sqrt(np.dot(w.T, w))
        w = w - stepSize*deltaL
        l = sum(np.log(1 + np.exp(-y * np.dot(x.T, w)))) + lamb*np.dot(w.T, w)
        print(l)

    return w


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
        y_train = np.append(y_train, -1)

    if i > len(os.listdir('./Data Sets/')) / 2:
        if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
            time, wave_data, noise_data = read_data(i)

        x_test = np.column_stack((x_test, wave_data))
        y_test = np.append(y_test, 1)
        x_test = np.column_stack((x_test, noise_data))
        y_test = np.append(y_test, -1)

x_train = (x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1)).T
x_test = (x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1)).T
y_train = y_train[1:]
y_test = y_test[1:]

n=100
trainErr = np.zeros(n)
testErr = np.zeros(n)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(x_train.T, y_train.ravel())

print('Time Series')
print('Training Accuracy: ', clf.score(x_train.T, y_train))
print('Testing Accuracy: ', clf.score(x_test.T, y_test))

print(clf.get_params())



x_train = np.zeros(129)
y_train = np.array(0)
x_test = np.zeros(129)
y_test = np.array(0)

for i in range(1, len(os.listdir('./Data Sets/'))):
    if i<=len(os.listdir('./Data Sets/'))/2:
        if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
            time, wave_data, noise_data = read_data(i)

        f, P1 = signal.welch(noise_data)
        f, P2 = signal.welch(wave_data)

        x_train = np.column_stack((x_train, np.log10(P2)))
        y_train = np.append(y_train, 1)
        x_train = np.column_stack((x_train, np.log10(P1)))
        y_train = np.append(y_train, -1)

    if i > len(os.listdir('./Data Sets/')) / 2:
        if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
            time, wave_data, noise_data = read_data(i)

        f, P1 = signal.welch(noise_data)
        f, P2 = signal.welch(wave_data)

        x_test = np.column_stack((x_test, np.log10(P2)))
        y_test = np.append(y_test, 1)
        x_test = np.column_stack((x_test, np.log10(P1)))
        y_test = np.append(y_test, -1)

x_train = (x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1)).T
x_test = (x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1)).T
y_train = y_train[1:]
y_test = y_test[1:]

# n=10
# trainErr = np.zeros(n)
# testErr = np.zeros(n)
#
# for i in range(n):
#     w = logReg(x_train, y_train, i)
#     trainErr[i] = 1 - sum(y_train == np.sign(np.dot(x_train.T,w))) / len(y_train)
#     testErr[i] = 1 - sum(y_test == np.sign(np.dot(x_test.T, w))) / len(y_train)
#     print(trainErr[i])
#     print(testErr[i])


clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train.T, y_train.ravel())

print('Spectrum')
print('Training Accuracy: ', clf.score(x_train.T, y_train))
print('Testing Accuracy: ', clf.score(x_test.T, y_test))

# plt.figure(1)
# plt.loglog(f, x_train)
# plt.draw()
#
# plt.figure(3)
# colors = itertools.cycle(["r", "b"])
# for i in range(len(x_train)):
#
#     plt.plot(x_train[51, i], x_train[121, i], '.', color=next(colors))
#
# plt.draw()
#
# plt.figure(2)
# plt.plot(range(n), trainErr)
# plt.plot(range(n), testErr)
# plt.draw()
#
# plt.show()
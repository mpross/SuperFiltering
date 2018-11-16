import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import itertools
import time
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA

def read_data(index):

    f = open('Data Sets/signal' + str(index) + '.dat', 'r')

    lines = f.read().split('\n')
    l = lines.__len__() - 1
    tim = np.zeros(l)
    wave_data = np.zeros(l)
    noise_data = np.zeros(l)

    for i in range(0, l):
        tim[i] = float(lines[i].split(' ')[0])
        wave_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    f = open('Data Sets/noise' + str(index) + '.dat', 'r')
    lines = f.read().split('\n')
    for i in range(0, l):
        noise_data[i] = float(lines[i].split(' ')[1])*10**23

    f.close()

    return tim, wave_data, noise_data


x_train = np.zeros(129)
y_train = np.array(0)
x_test = np.zeros(129)
y_test = np.array(0)

for i in range(1, len(os.listdir('./Data Sets/'))):
    if i<=len(os.listdir('./Data Sets/'))/2:
        if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
            tim, wave_data, noise_data = read_data(i)

        f, P1 = signal.welch(noise_data)
        f, P2 = signal.welch(wave_data)

        x_train = np.column_stack((x_train, np.log10(P2)))
        y_train = np.append(y_train, 1)
        x_train = np.column_stack((x_train, np.log10(P1)))
        y_train = np.append(y_train, -1)

    if i > len(os.listdir('./Data Sets/')) / 2:
        if os.path.isfile('Data Sets/signal' + str(i) + '.dat'):
            tim, wave_data, noise_data = read_data(i)

        f, P1 = signal.welch(noise_data)
        f, P2 = signal.welch(wave_data)

        x_test = np.column_stack((x_test, np.log10(P2)))
        y_test = np.append(y_test, 1)
        x_test = np.column_stack((x_test, np.log10(P1)))
        y_test = np.append(y_test, -1)


def svd_plot(alg, index, name):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = alg.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    colors = itertools.cycle(["r", "b"])

    plt.figure(index)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    for i in range(len(x_train)):
        plt.plot(pca.transform(x_train)[i, 0], pca.transform(x_train)[i, 1], '.', color=next(colors))

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(name)
    plt.draw()


x_train = (x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1))
x_test = (x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1))
y_train = y_train[1:]
y_test = y_test[1:]

start = time.time()

logReg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train, y_train.ravel())

trainTime = time.time() - start
start = time.time()

print('Logistic Regression')
print('Training Accuracy: ', logReg.score(x_train, y_train))
print('Testing Accuracy: ', logReg.score(x_test, y_test))
print('Training Time: ', trainTime, ' s')
print('Execution Time: ', time.time()-start, ' s')

start = time.time()

svmAlg = svm.SVC(gamma='scale').fit(x_train, y_train.ravel())

trainTime = time.time() - start
start = time.time()

print('SVM')
print('Training Accuracy: ', svmAlg.score(x_train, y_train))
print('Testing Accuracy: ', svmAlg.score(x_test, y_test))
print('Training Time: ', trainTime, ' s')
print('Execution Time: ', time.time()-start, ' s')


pca = PCA(n_components=2).fit(x_train)
X = pca.transform(x_train)

logReg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y_train.ravel())
svmAlg = svm.SVC(gamma='scale').fit(X, y_train.ravel())

svd_plot(logReg, 1, 'Logistic Regression')
svd_plot(svmAlg, 2, 'SVM')

plt.show()
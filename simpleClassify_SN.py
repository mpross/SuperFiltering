import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy import signal
import os
import itertools
import time
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



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
            wave_data[i] = float(lines[i].split(' ')[1])*10**25

    f.close()

    f = open('Supernova Data/Gain'+str(gain)+'/noise' + str(index) + '.dat', 'r')
    lines = f.read().split('\n')
    l = lines.__len__() - 1
    for i in range(0, l):
        if not(np.isnan(float(lines[i].split(' ')[1]))):
            noise_data[i] = float(lines[i].split(' ')[1])*10**25

    f.close()

    return tim, wave_data, noise_data


gainList = np.array((0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 ,0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10))

NNAcc = np.zeros(gainList.size)
NearNAcc = np.zeros(gainList.size)
logRegAcc = np.zeros(gainList.size)
SVMAcc = np.zeros(gainList.size)

gainIndex = 0

for gain in gainList:

	print(str(gainList[gainIndex]))
	#x_train = np.zeros(129)
	x_train = np.zeros(2049)
	y_train = np.array(0)
	#x_test = np.zeros(129)
	x_test = np.zeros(2049)
	y_test = np.array(0)

	for i in range(1, len(os.listdir('./Supernova Data/Gain'+str(gain)+'/'))):
		if i<=len(os.listdir('./Supernova Data/Gain'+str(gain)+'/'))/2:
			if os.path.isfile('Supernova Data/Gain'+str(gain)+'/signal' + str(i) + '.dat') & \
				os.path.isfile('Supernova Data/Gain'+str(gain)+'/noise' + str(i) + '.dat'):
					tim, wave_data, noise_data = read_data(i, gain)

		f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096)
		f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096)

		with np.errstate(divide='raise'):
			try:
				x_train = np.column_stack((x_train, np.log10(P2)))
				y_train = np.append(y_train, 1)
				x_train = np.column_stack((x_train, np.log10(P1)))
				y_train = np.append(y_train, -1)
			except FloatingPointError:
				print('Error skipping this data point')

		if i > len(os.listdir('./Supernova Data/Gain'+str(gain)+'/')) / 2:
			if os.path.isfile('Supernova Data/Gain'+str(gain)+'/signal' + str(i) + '.dat') & \
				os.path.isfile('Supernova Data/Gain'+str(gain)+'/noise' + str(i) + '.dat'):
					tim, wave_data, noise_data = read_data(i, gain)

		f, P1 = signal.welch(noise_data, fs=4096, nperseg=4096)
		f, P2 = signal.welch(wave_data, fs=4096, nperseg=4096)

		with np.errstate(divide='raise'):
			try:
				x_test = np.column_stack((x_test, np.log10(P2)))
				y_test = np.append(y_test, 1)
				x_test = np.column_stack((x_test, np.log10(P1)))
				y_test = np.append(y_test, -1)
			except FloatingPointError:
				print('Error skipping this data point')


	x_train = (x_train[:, 1:].T-np.mean(x_train[:, 1:], axis=1))
	x_test = (x_test[:, 1:].T-np.mean(x_test[:, 1:], axis=1))
	y_train = y_train[1:]
	y_test = y_test[1:]
	try:
		start = time.time()

		logReg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000)
		logReg.fit(x_train, y_train.ravel())

		trainTime = time.time() - start
		start = time.time()
	
        # print('Logistic Regression')
        # print('Training Accuracy: ', logReg.score(x_train, y_train))
        # print('Testing Accuracy: ', logReg.score(x_test, y_test))
        # print('Training Time: ', trainTime, ' s')
        # print('Execution Time: ', time.time()-start, ' s')

		logRegAcc[gainIndex] = logReg.score(x_test, y_test)


		start = time.time()

		svmAlg = svm.SVC(gamma='scale')

		svmAlg.fit(x_train, y_train.ravel())

		trainTime = time.time() - start
		start = time.time()

        # print('SVM')
        # print('Training Accuracy: ', svmAlg.score(x_train, y_train))
        # print('Testing Accuracy: ', svmAlg.score(x_test, y_test))
        # print('Training Time: ', trainTime, ' s')
        # print('Execution Time: ', time.time()-start, ' s')

		SVMAcc[gainIndex] = svmAlg.score(x_test, y_test)

		start = time.time()

		NearN = KNeighborsClassifier(10)
		NearN.fit(x_train, y_train.ravel())

		trainTime = time.time() - start
		start = time.time()

        # print('Nearest Neighbors')
        # print('Training Accuracy: ', NearN.score(x_train, y_train))
        # print('Testing Accuracy: ', NearN.score(x_test, y_test))
        # print('Training Time: ', trainTime, ' s')
        # print('Execution Time: ', time.time()-start, ' s')

		NearNAcc[gainIndex] = NearN.score(x_test, y_test)

		start = time.time()

		NN = MLPClassifier(max_iter=10000)
		NN.fit(x_train, y_train.ravel())

		trainTime = time.time() - start
		start = time.time()

        # print('Neural Network')
        # print('Training Accuracy: ', NN.score(x_train, y_train))
        # print('Testing Accuracy: ', NN.score(x_test, y_test))
        # print('Training Time: ', trainTime, ' s')
        # print('Execution Time: ', time.time()-start, ' s')

		NNAcc[gainIndex] = NN.score(x_test, y_test)
	except ValueError:
		print('Error')

	gainIndex += 1

plt.figure(10)
plt.plot(gainList, logRegAcc)
plt.plot(gainList, SVMAcc)
plt.plot(gainList, NearNAcc)
plt.plot(gainList, NNAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Gain')
plt.legend(('Logistic Regression', 'SVM', 'Nearest Neighbor', 'Neural Network'))
plt.savefig('SimpleAccuracySN.pdf')

plt.figure(11)
plt.plot(10/gainList, logRegAcc)
plt.plot(10/gainList, SVMAcc)
plt.plot(10/gainList, NearNAcc)
plt.plot(10/gainList, NNAcc)
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('Distance (kpc)')
plt.legend(('Logistic Regression', 'SVM', 'Nearest Neighbor', 'Neural Network'))
plt.savefig('SimpleAccuracyDistanceSN.pdf')


#
# pca = PCA(n_components=2).fit(x_train)
# X = pca.transform(x_train)
#
# logReg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y_train.ravel())
# svmAlg = svm.SVC(gamma='scale').fit(X, y_train.ravel())
# NearN = KNeighborsClassifier(1).fit(X, y_train.ravel())
# NN = MLPClassifier(max_iter=1000).fit(X, y_train.ravel())
#
# svd_plot(logReg, 1, 'Logistic Regression')
# svd_plot(svmAlg, 2, 'SVM')
# svd_plot(NearN, 3, 'Nearest Neighbors')
# svd_plot(NN, 4, 'Neural Network')
#
plt.show()

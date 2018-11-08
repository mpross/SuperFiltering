import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import listdir


def read_LIGO_data():

    filename = 'LIGO Data/H-H1_LOSC_4_V1-1135210496-4096.hdf5'
    f = h5py.File(filename, 'r')

    data = np.array(f['strain']['Strain'])

    time = np.array(range(data.size))*list(f['strain']['Strain'].attrs.values())[3]
    print(time.size)

    return time, data


def read_waveform():
    path = 'Supernova Waveforms'
    ignore_list = ['README_signal', 'signal_data.tar.gz']
    files_list = [f for f in listdir(path) if not(f in ignore_list)]

    index=np.random.randint(0,files_list.__len__())
    print(index)

    f = open(path+'/'+files_list[index])
    fin = f.read()
    fin = fin.split('\n')
    fsize = fin.__len__()

    time = np.zeros(fsize)
    data = np.zeros(fsize)
    for i in range(fsize):
        try:
            time[i] = float(fin[i].split(' ')[0])
            data[i] = float(fin[i].split(' ')[1])
        except ValueError:
            time[i] = 0
            data[i] = 0

    f.close()

    return time, data


waveform_time, waveform_data = read_waveform()

LIGO_time, LIGO_data = read_LIGO_data()

plt.plot(LIGO_time, LIGO_data)
plt.show()
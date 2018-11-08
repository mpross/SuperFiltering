inf=hdf5info('LIGO Data/H-H1_LOSC_4_V1-1135210496-4096.hdf5');
LIGOraw=hdf5read('LIGO Data/H-H1_LOSC_4_V1-1135210496-4096.hdf5',inf.GroupHierarchy.Groups(3).Datasets(1).Attributes(1).Location);
sampF=1/(inf.GroupHierarchy.Groups(3).Datasets.Attributes(4).Value);

SuperRaw=load('Supernova Waveforms/signal_e15a_ls.dat');

super=decimate(SuperRaw(:,2),floor(1/(sampF*mean(diff(SuperRaw(:,1))/1000))));

time=(1:length(super))/sampF;
[b,a]=butter(3,10/sampF,'high');
strain=filter(b,a,LIGOraw(1:length(super)))+super*10;

[A,F]=asd2(LIGOraw,1/sampF,21,1,@hann);
[A2,F2]=asd2(strain,1/sampF,3,1,@hann);

figure(1)
plot(time,strain)
xlim([0, length(super)/sampF])

figure(2)
loglog(F,A,F2,A2)
xlim([10 1e3])


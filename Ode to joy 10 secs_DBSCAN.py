#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the pyplot and wavfile modules 
import matplotlib.pyplot as plot
from scipy.io import wavfile
import matplotlib 
import numpy as np
from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.cluster import DBSCAN
import statistics


lowFreq = 150
maxFreq = 550
timescaling = 25
nfft = 22050
noverlap = 20050


# In[3]:


vln_freq = np.array([196,207.65,220,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.30,440,466.16,493.88,523.25,554.37])
vln_notes = np.array(['G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5'])


# In[16]:


# Read the wav file (mono)
samplingFrequency, signalData = wavfile.read(r'C:\Users\tason\OneDrive\Desktop\ITSP\violin_ode to joy.wav')
# samplingFrequency, signalData = wavfile.read(r'C:\Users\tason\OneDrive\Desktop\ITSP\Violin_For He is A Jolly Good Fellow.wav')

Pxx, freq, bins, im = plot.specgram(signalData[0:441000,0],NFFT=22050,noverlap=20050,Fs=samplingFrequency)#,cmap='jet')
# Pxx, freq, bins, im = plot.specgram(signalData[:,0],NFFT=nfft ,noverlap=noverlap,Fs=samplingFrequency)#,cmap='jet')

plot.xlabel('Time')
plot.ylabel('Frequency')
plot.ylim([lowFreq,maxFreq])
plot.rcParams["figure.figsize"] = (12.8, 15)
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
plot.locator_params(axis="y", nbins=16)
plot.locator_params(axis="x", nbins=16)
plot.colorbar()
#############################################################
l_index = np.argmin(abs(freq-lowFreq))
m_index = np.argmin(abs(freq-maxFreq))

for i in range(bins.size):
    max_value = np.max(Pxx[l_index:m_index,i])
    max_index = np.argmax(Pxx[l_index:m_index,i]) + l_index
    max_freq = freq[max_index]
    plot.plot(bins[i],max_freq,'r.')
    
# Forming array of concerned frequencies
con_freq=[]
for i in range(bins.size):
    max_value = np.max(Pxx[l_index:m_index,i])
    max_index = np.argmax(Pxx[l_index:m_index,i]) + l_index
    max_freq = freq[max_index]
    con_freq.append(max_freq)
    
X=np.empty([bins.size,2])
X[:,0]=bins*timescaling 
X[:,1]=con_freq


# In[17]:


# define the model
dbscan_model = DBSCAN(eps=3, min_samples=5)

# train the model
dbscan_model.fit(X)

# assign each data point to a cluster
dbscan_result = dbscan_model.fit_predict(X)

# get all of the unique clusters
dbscan_clusters = unique(dbscan_result)


# plot the DBSCAN clusters
clt_bin=[]
clt_freq=[]
for dbscan_cluster in dbscan_clusters:
    if dbscan_cluster!=-1:
        # get data points that fall in this cluster
        idx = where( dbscan_result == dbscan_cluster )
        cltbin = np.average(X[idx,0])
        cltfreq = np.average(X[idx,1])  
        clt_bin.append(cltbin)
        clt_freq.append(cltfreq)
        plot.plot(cltbin/timescaling ,cltfreq,'k*',markersize=15)
# print(clt_freq,clt_bin)
    
    
# make the plot
plot.scatter(X[:,0]/timescaling , X[:,1],c=dbscan_result, cmap='Paired')
# show the DBSCAN plot
# pyplot.show()

# type(idx)
         
plot.specgram(signalData[0:441000,0],NFFT=nfft,noverlap=noverlap,Fs=samplingFrequency)#,cmap='jet')
plot.ylim([lowFreq,maxFreq])


# In[18]:


fin_freq = clt_freq
len(fin_freq)


# In[19]:


fin_note = []
for i in fin_freq:
    fin_note.append(vln_notes[np.argmin(abs(vln_freq-i))])
    
print(fin_note)    


# In[ ]:





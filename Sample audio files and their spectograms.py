#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the pyplot and wavfile modules 
import matplotlib.pyplot as plot
from scipy.io import wavfile
import matplotlib 
import numpy as np


# In[14]:


# Read the wav file (mono)
samplingFrequency, signalData_1 = wavfile.read(r'C:\Users\tason\OneDrive\Desktop\ITSP\Happy birthday_piano.wav')
samplingFrequency, signalData_2 = wavfile.read(r'C:\Users\tason\OneDrive\Desktop\ITSP\Happy Birthday_C scale Flute.wav')


# In[18]:


# Plot the signal read from wav file
plot.subplot(211)
plot.title('Signal of a wav file with piano music')
plot.plot(signalData_1[:,1])
plot.xlabel('Sample')
plot.ylabel('Amplitude')
print(signalData_1.shape)


# In[26]:


plot.subplot(212)
Pxx, freq, bins, im = plot.specgram(signalData_1[0:1014300,0],NFFT=22050,noverlap=20000,Fs=samplingFrequency)#,cmap='jet')
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.ylim([350,850])
plot.rcParams["figure.figsize"] = (12.8, 15)
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
plot.locator_params(axis="y", nbins=16)
plot.locator_params(axis="x", nbins=16)
plot.colorbar()

print(Pxx.shape)
print(freq.shape)
print(bins.shape)


# In[20]:


# Plot the signal read from wav file
plot.subplot(211)
plot.title('Signal of a wav file with Flute music')
plot.plot(signalData_2[:,1])
plot.xlabel('Sample')
plot.ylabel('Amplitude')
print(signalData_2.shape)


# In[24]:


plot.subplot(212)
Pxx, freq, bins, im = plot.specgram(signalData_2[0:1014300,0],NFFT=22050,noverlap=20050,Fs=samplingFrequency)#,cmap='jet')
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.ylim([0,1000])
plot.rcParams["figure.figsize"] = (12.8, 15)
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
plot.locator_params(axis="y", nbins=16)
plot.locator_params(axis="x", nbins=16)
plot.colorbar()

print(Pxx.shape)
print(freq.shape)
print(bins.shape)


# In[ ]:





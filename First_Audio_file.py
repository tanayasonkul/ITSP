#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the pyplot and wavfile modules 
import matplotlib.pyplot as plot
from scipy.io import wavfile
import matplotlib 
import numpy as np


# In[2]:


# Read the wav file (mono)
samplingFrequency, signalData = wavfile.read(r'C:\Users\tason\OneDrive\Desktop\ITSP\Happy birthday violin.wav')


# In[3]:


# Plot the signal read from wav file
plot.subplot(211)
plot.title('Signal of a wav file with piano music')
plot.plot(signalData[:,1])
plot.xlabel('Sample')
plot.ylabel('Amplitude')
print(signalData.shape)


# In[16]:


plot.subplot(212)
Pxx, freq, bins, im = plot.specgram(signalData[0:1014300,0],NFFT=22050,noverlap=20050,Fs=samplingFrequency)#,cmap='jet')
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.ylim([150,580])
plot.rcParams["figure.figsize"] = (12.8, 15)
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
plot.locator_params(axis="y", nbins=16)
plot.locator_params(axis="x", nbins=16)
plot.colorbar()

print(Pxx.shape)
print(freq.shape)
print(bins.shape)

l_index = np.argmin(abs(freq-150))
m_index = np.argmin(abs(freq-580))


for i in range(bins.size):
    max_value = np.max(Pxx[l_index:m_index,i])
    max_index = np.argmax(Pxx[l_index:m_index,i]) + l_index
    max_freq = freq[max_index]
#     print(max_freq)
    plot.plot(bins[i],max_freq,'r.')
    

# for i in range(bins.size):
#     max_value = np.max(Pxx[:,i])
#     max_index = np.argmax(Pxx[:,i])
#     max_freq = freq[max_index]
#     print(max_value)
#     plot.plot(bins[i],max_freq,'r.')


plot.plot(kmeans.cluster_centers_[:,0]/20,kmeans.cluster_centers_[:,1],'g.')


# In[5]:


print(samplingFrequency)


# In[ ]:





# In[6]:


# getting 
max_value = np.max(Pxx[:,0])
max_index = np.argmax(Pxx[:,0])
max_freq = freq[max_index]
print(max_value,max_index,max_freq)
# plot.plot(bins[0],max_freq,'r')

l_index = np.argmin(abs(freq-150))
m_index = np.argmin(abs(freq-580))
print(l_index,m_index)


# In[7]:


# Forming array of concerned frequencies
con_freq=[]
for i in range(bins.size):
    max_value = np.max(Pxx[l_index:m_index,i])
    max_index = np.argmax(Pxx[l_index:m_index,i]) + l_index
    max_freq = freq[max_index]
    con_freq.append(max_freq)
    print(max_freq)
print(con_freq)


# In[8]:


plot.plot(bins,con_freq,'.')


# In[9]:


X=np.empty([bins.size,2])
X[:,0]=bins*20
X[:,1]=con_freq


# In[10]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=28, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
# plot.scatter(X[:,0], X[:,1])
# plot.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# plot.show()
kmeans.cluster_centers_.shape
# plot.plot(kmeans.cluster_centers_,'g.')


# In[11]:


# sort_clt = np.sort(kmeans.cluster_centers_, axis=0) 
# np.argsort(sort_clt)
clt_freq= kmeans.cluster_centers_[:, 1]
fin_freq = clt_freq[np.argsort(kmeans.cluster_centers_[:, 0])]
# fin_freq = sort_clt[:,1]
print(fin_freq)


# In[12]:


vln_freq = np.array([196,207.65,220,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.30,440,466.16,493.88,523.25,554.37])
vln_notes = np.array(['G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5'])
type(vln_notes)
vln_freq.size
print(np.diff(vln_freq))


# In[13]:


fin_note = []
for i in fin_freq:
    fin_note.append(vln_notes[np.argmin(abs(vln_freq-i))])
    
print(fin_note)    


# In[14]:


estimator = KMeans(n_clusters=25)
estimator.fit(X)
nclt=np.zeros([25])
for i in estimator.labels_:
    nclt[i]=nclt[i]+1
# print(nclt,estimator.labels_)
plot.plot(np.sort(nclt),'.')
# np.std(nclt)


# In[15]:


# Read image 
import cv2
img = cv2.imread('Task 1.png', cv2.IMREAD_COLOR) # road.png is the filename
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=10, maxLineGap=250)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
cv2.imshow("Result Image", img)


# In[5]:


import cv2
import numpy as np

img = cv2.imread('Task 1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray,100,200,apertureSize = 3)
cv2.imshow('edges',edges)
cv2.waitKey(0)

minLineLength = 1000
maxLineGap = 1000
lines = cv2.HoughLinesP(edges,1,np.pi/180,1000,minLineLength=minLineLength,maxLineGap=maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('hough',img)
cv2.waitKey(0)


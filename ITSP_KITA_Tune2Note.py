#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'> Importing Libraries </font> #

# In[2]:


import matplotlib.pyplot as plot
from scipy.io import wavfile
import matplotlib 
import numpy as np
from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.cluster import DBSCAN
import statistics
from scipy.signal import find_peaks,peak_widths
from fpdf import FPDF


# # <font color='blue'> Defining Violin Notes and corresponding Frequencies </font> #

# In[15]:


vln_freq = np.array([196,207.65,220,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.30,440,466.16,493.88,523.25,554.37,587.33,622.25,659.25])
vln_notes = np.array(['G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5','D5','D#5','E5'])


# # <font color='blue'> Defining Spectrogram and other constant parameters </font> #

# In[4]:


lowFreq = 150
maxFreq = 550
timescaling = 25
nfft = 22050
noverlap = 20050
fs = 44100


# # <font color='blue'> Function to get frequencies in spectrogram with high energies </font> #

# In[5]:


def get_con_freq(Pxx, freq, bins):
#     plot.xlabel('Time')
#     plot.ylabel('Frequency')
#     plot.ylim([lowFreq,maxFreq])
#     plot.rcParams["figure.figsize"] = (12.8, 15)
#     matplotlib.rc('xtick', labelsize=15) 
#     matplotlib.rc('ytick', labelsize=15) 
#     plot.locator_params(axis="y", nbins=16)
#     plot.locator_params(axis="x", nbins=16)
#     plot.colorbar()
    #############################################################
    l_index = np.argmin(abs(freq-lowFreq))
    m_index = np.argmin(abs(freq-maxFreq))

    con_freq=[]
    # temp_1=[]
    for i in range(bins.size):
        max_value = np.max(Pxx[l_index:m_index,i])
    #     print((max_value))
    #     temp_1.append(max_value)
        max_index = np.argmax(Pxx[l_index:m_index,i]) + l_index
        max_freq = freq[max_index]
        con_freq.append(max_freq)
    #     plot.plot(bins[i],max_freq,'r.')

    X=np.empty([bins.size,2])
    X[:,0]=bins*timescaling 
    X[:,1]=con_freq

    return X;


# # <font color='blue'> Function to cluster high energy points with DBSCAN Cluster algo.  </font> #

# In[6]:



def get_clt(X):

    # define the model
    dbscan_model = DBSCAN(eps=5, min_samples=6)


    # train the model
    dbscan_model.fit(X)

    # assign each data point to a cluster
    dbscan_result = dbscan_model.fit_predict(X)
    # print(dbscan_result)

    # get all of the unique clusters
    dbscan_clusters = unique(dbscan_result)

    #No. of clusters
    print((dbscan_clusters.size)-1)

    # plot the DBSCAN clusters
    clt_bin=[]
    clt_freq=[]
    clt_pnt=[]
    clt_time=[]
    clt_idx = []
    for dbscan_cluster in dbscan_clusters:
        idx = where( dbscan_result == dbscan_cluster )

        if dbscan_cluster!=-1:
    #         print(idx)
            clt_idx.append(idx)
            clt_time.append(bins[max(idx[0])]-bins[min(idx[0])])
            cltbin = np.average(X[idx,0])
            cltfreq = np.average(X[idx,1]) 
            clt_pnt.append(len(idx[0]))
            clt_bin.append(cltbin)
            clt_freq.append(cltfreq)
    #         plot.plot(cltbin/timescaling ,cltfreq,'k*',markersize=15)
    # print(clt_freq,clt_bin)


    # make the plot
    # plot.scatter(X[:,0]/timescaling , X[:,1],c=dbscan_result, cmap='Paired')



    # plot.specgram(signalData[6*fs:50*fs,0],NFFT=nfft,noverlap=noverlap,Fs=samplingFrequency)#,cmap='jet')
    # plot.specgram(signalData,NFFT=nfft,noverlap=noverlap,Fs=samplingFrequency)#,cmap='jet')
    # plot.ylim([lowFreq,maxFreq])

    return dbscan_clusters,clt_idx,clt_freq;


# print (len(clt_freq))



# ## <font color='blue'> Function to detect number of same consecutive notes if any in detected frequency clusters </font> ##

# In[21]:




# plot.rcParams["figure.figsize"] = (12.8/4, 15/4)


def get_pks(dbscan_clusters,clt_idx, clt_freq, freq):
    l_index = np.argmin(abs(freq-lowFreq))
    m_index = np.argmin(abs(freq-maxFreq))

    pks = []
#     for i in range((dbscan_clusters.size)-1):
    for i in range(len(clt_idx)):

        temp = []
        for j in range (min((clt_idx[i])[0]),max((clt_idx[i])[0])):
            max_val = np.max(Pxx[l_index:m_index,j])
            temp.append(max_val)


        #############
        x = np.asarray(temp)
        peaks, properties = find_peaks(x, prominence=1, width=1)
#         print("cluster no.:",i+1," Peaks: ",peaks)
    #     print(type(peaks))
        if (len(peaks) > 1):
#             plot.plot(x)
#             plot.plot(peaks, x[peaks], "x")
#             plot.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],ymax = x[peaks], color = "C1")
#             plot.hlines(y=properties["width_heights"], xmin=properties["left_ips"],xmax=properties["right_ips"], color = "C2")
    #         print("Prom/width :",properties["prominences"]/properties["width_heights"])
    #         print(len(properties["prominences"]))
    #         print("Prom.prom_max:",properties["prominences"]/max(properties["prominences"])  )
    #         print("ratio_per:",ratio_per)


            ratio = properties["prominences"]/properties["width_heights"]
            ratio_per = ratio/max(ratio)
    #         print(type(ratio))
            prom_ratio = properties["prominences"]/max(properties["prominences"])

#             print("Prom/width :",ratio)
#              print(len(properties["prominences"]))
#             print("Prom.prom_max:",prom_ratio  )
#             print("ratio_per:",ratio_per)

#             plot.show() 

            ratio_ind = np.where(ratio > 0.25)
            prom_ind = np.where(prom_ratio > 0.1)

            ratio_1 = ratio[ratio_ind]
            prom_ratio_1 =  prom_ratio[prom_ind]

            pks_1 = len(ratio_1)
            pks_2 = len(prom_ratio_1)

            pks.append(min(pks_1,pks_2)) 
        else:    
#             plot.plot(x)
#             plot.plot(peaks, x[peaks], "x")
#             plot.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],ymax = x[peaks], color = "C1")
#             plot.hlines(y=properties["width_heights"], xmin=properties["left_ips"],xmax=properties["right_ips"], color = "C2")
#             plot.show() 

            pks.append(1)

            return clt_freq, pks;
    
        
        





# # <font color='blue'> Function to convert frequencies into notes  </font> #

# In[22]:


# vln_freq = np.array([196,207.65,220,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.30,440,466.16,493.88,523.25,554.37,587.33,622.25,659.25])
# vln_notes = np.array(['G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5','D5','D#5','E5'])


def get_fin_note(clt_freq, pks, vln_freq,vln_notes):
#     fin_freq = clt_freq
#     print(len(clt_freq))

#     vln_freq = np.array([196,207.65,220,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.30,440,466.16,493.88,523.25,554.37,587.33,622.25,659.25])
#     vln_notes = np.array(['G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5','D5','D#5','E5'])

    fin_note = []
    for i in range(len(clt_freq)):
        fin_note += pks[i]*[(vln_notes[np.argmin(abs(vln_freq-clt_freq[i]))])]

    print(len(fin_note)) 
    return fin_note;

# print(len(fin_note))


# # <font color='blue'> Output a pdf with obtained notes for given/uploaded song </font> #

# In[23]:


def get_pdf(fin_note):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=fin_note, ln=1, align="C")

    for i in fin_note:
        pdf.write(5,str(i))
        pdf.write(5,'  ')

    #     pdf.ln()
    pdf.output("Notes.pdf")
    return;


# In[24]:


# vln_freq = np.array([196,207.65,220,233.08,246.94,261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.30,440,466.16,493.88,523.25,554.37,587.33,622.25,659.25])
# vln_notes = np.array(['G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5','D5','D#5','E5'])


samplingFrequency, signalData = wavfile.read(r'C:/Users/tason/OneDrive/Desktop/ITSP/Ode to Joy on the violin.wav')
Pxx, freq, bins, im = plot.specgram(signalData[:,0],NFFT=nfft,noverlap=noverlap,Fs=samplingFrequency)#,cmap='jet')

X = get_con_freq(Pxx, freq, bins)
dbscan_clusters,clt_idx,clt_freq = get_clt(X)
clt_freq, pks = get_pks(dbscan_clusters,clt_idx,clt_freq,freq)
fin_note = get_fin_note(clt_freq, pks,vln_freq,vln_notes)
print(len(fin_note)) 

get_pdf(fin_note)


# In[ ]:





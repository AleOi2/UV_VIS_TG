#!usr/bin/python


import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os
import seaborn as sns
import codecs

from pylab import rcParams
from Absorbance import *

# Set Plotting parameters
sns.set_style('ticks', {'xtick.direction':u'in', 'ytick.direction':u'in', 'image.cmap': u'coolwarm'})

rcParams['figure.figsize'] = 10, 8



class Recoder(object):
    # This class recodes UTF-8 and UTF-16 CSV files
    def __init__(self, stream, decoder, encoder, eol='\r\n'):
        self._stream = stream
        self._decoder = decoder if isinstance(decoder, codecs.IncrementalDecoder) else codecs.getincrementaldecoder(decoder)()
        self._encoder = encoder if isinstance(encoder, codecs.IncrementalEncoder) else codecs.getincrementalencoder(encoder)()
        self._buf = ''
        self._eol = eol
        self._reachedEof = False
    
    def read(self, size=None):
        r = self._stream.read(size)
        raw = self._decoder.decode(r, size is None)
        return self._encoder.encode(raw)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._reachedEof:
            raise StopIteration()
        while True:
            line,eol,rest = self._buf.partition(self._eol)
            if eol == self._eol:
                self._buf = rest
                return self._encoder.encode(line + eol)
            raw = self._stream.read(1024)
            if raw == '':
                self._decoder.decode(b'', True)
                self._reachedEof = True
                return self._encoder.encode(self._buf)
            self._buf += self._decoder.decode(raw)
    next = __next__
    
    def close(self):
        return self._stream.close()

def Extract_Data(File_List, Range):
    # Function for extracting data
    N = len(File_List)
    AU_List = []
    # Loop through each file and store the absorbance data in an absorbance object
    for File in File_List:
        Name = File.split('_')[0]
        Temp = int(File.split('_')[1])
        WL = []
        AU = []
        with open(File, 'r') as csvfile:
            csvfile = Recoder(csvfile, 'utf-16', 'utf-8')
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                try:
                    WL_Temp = int(row[0])
                    AU_Temp = float(row[1])
                    WL.append(WL_Temp)
                    AU.append(AU_Temp)
                except:
                    continue

        AU = np.asarray(AU[Range[0]:Range[1]])
        WL = np.asarray(WL[Range[0]:Range[1]])
        A = Absorbance(Name, Temp, AU, WL)
        A.Smooth()
        A.Normalize()
        AU_List.append(A)
        if Temp == 25:
            AU_List[-1].As_Cast = True
    return AU_List


def Plot_UV(AU_List):
    #Function for plotting UV Vis
    for A in AU_List:
        A.Plot_Info()
    plt.show()
    return


def Fit_Lines(T,A):
    # Fits a bunch of bilinear regressions to the data and finds the best one.
    T = np.asarray(T) # Converts Temp to numpy array
    A = np.asarray(A) # Converts Absorb to numpy array
    N = len(T)
    res = []
    m = 1
    n = 2 # Outer points to exclude from fitting
    for i in range(n, N-n):
        m1, b1, r1, p1, std1= stats.linregress(T[0:i], A[0:i]) # Fit Glassy regime
        m2, b2, r2, p2, std2 = stats.linregress(T[i+m:N], A[i+m:N]) # Fit Melt regime
        Tg = (b1 - b2)/(m2 -m1) # Compute Tg
        # Compute sum of errors from bilinear regression
        Weighted_Res = (i*r1**2 + (N-i)*r2**2) # Each regression weighted based on number of points it contains
        res.append(Weighted_Res)
        
        print Tg, res[-1]
        plt.plot(T, A, linestyle = '-', marker='o', color='k')
        plt.ylim((A.min()-0.01, A.max() + 0.01))
        plt.plot(T, m1*T + b1, 'r', label = "Tg = %d --> R$^2$ = %.2f" % (Tg, res[-1]))
        plt.plot(T, m2*T + b2, 'r')
        plt.xlabel("Annealing Temperature ($^\circ$C)", fontsize=25)
        plt.ylabel("Deviation Metric", fontsize=25)
        plt.legend()
    
    res_array = np.asarray(res)
    Inc =  np.argmax(res_array) + n  # Find index for maximum value of R2
    
    print res
    print Inc
    # Plot Best fit with all the other fits
    m1, b1, r1, p1, std1= stats.linregress(T[0:Inc], A[0:Inc])
    m2, b2, r2, p2, std2 = stats.linregress(T[Inc+m:N], A[Inc+m:N])
    Tg = (b1 - b2)/(m2 -m1)
    plt.plot(T, m1*T+ b1, 'g', linewidth=3)
    plt.plot(T, m2*T + b2, 'g', linewidth=3, label = '%d' % Tg)
    plt.legend(loc='upper left', frameon=False, fontsize=25)
    plt.tick_params( labelsize = 20, width=2, length=7)
    print Tg
    plt.show()
    return Inc, m1, b1, m2, b2, Tg


def Plot_DM(AU_List):
    # Compute the deviation metric for the absorbance data from each annealed film.
    if AU_List[0].As_Cast:
        print "Found As Cast"
    else:
        print "No As-Cast film. Please label As-Cast as T=25C and rerun"
        return
    MSD = []
    Temp = []
    for i in range(1,len(AU_List)):
        MSD_Temp = np.sqrt(np.sum((AU_List[0].N_AU_5M - AU_List[i].N_AU_5M)**2))
        MSD_Temp = MSD_Temp**2 # TBD
        MSD.append(MSD_Temp)
        Temp.append(AU_List[i].Temp_F)
    Temp = np.asarray(Temp)
    Temp_F = Temp*0.9833 + 0.7451
    MSD = np.asarray(MSD)

    # Plot deviation metric
    Inc, m1, b1, m2, b2, Tg = Fit_Lines(Temp_F,MSD)
    plt.plot(Temp_F[0:Inc+3], m1*Temp_F[0:Inc+3] + b1, 'r', linewidth=5)
    plt.plot(Temp_F, m2*Temp_F + b2, 'r', linewidth=5)
    plt.plot(Temp_F, MSD, 'o', markersize=10, color='k')
    plt.ylim((0, max(MSD)))
    plt.tick_params( labelsize = 20, width=2, length=7)
    plt.ylabel('Deviation Metric', fontsize=25)
    plt.xlabel('Annealing Temperature ($^\circ$C)', fontsize=25)
    plt.legend(loc='upper right', frameon=False, fontsize=25)
    plt.axvline(x=Tg,linewidth=4, color='g', alpha=0.5);
    plt.show()
    return


def Prepare_File_List( Name):
    # Function for reading the data and sorting based on temperature.
    File_List = glob.glob("%s_*.CSV" % Name )
    X= []

    for File in File_List:
        X.append(int(File.split("_")[1]))
        Index = File.split("_")[2]

    X = sorted(X)
    File_List2 = []

    for Value in X:
        File_List2.append("%s_%d_%s" % (Name, Value, Index))
    # rcParams['axes.color_cycle']= sns.color_palette('coolwarm', len(File_List2))

    return File_List2
    

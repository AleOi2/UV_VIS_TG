#!usr/bin/python

import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import codecs
import os
import seaborn as sns

class Recoder(object):
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


sns.set_style('ticks', {'xtick.direction':u'in', 'ytick.direction':u'in', 'image.cmap': u'coolwarm'})




class Absorbance:
    # Class for storing data
    def __init__(self, Name, Temp, AU, WL):
        self.Name = Name
        self.Temp = Temp
        self.AU = np.asarray(AU)
        self.WL = WL
        self.Ev = 1239.8/self.WL
        self.As_Cast = False
        self.N_AU  = self.AU/max(self.AU)
        self.N_5M_AU = np.empty_like(self.N_AU)
        for i in range(0,len(self.N_AU)-5):
            self.N_5M_AU[i] = (self.N_AU[i] + self.N_AU[i+1] + self.N_AU[i+2] + self.N_AU[i+3] + self.N_AU[i+4])/5.0
        
        return

    def Print_Info(self):
        print self.Name, self.Temp
        return

    def Plot_Info(self):
        self.N_AU = self.AU/max(self.AU)
        print max(self.AU)
        plt.plot(self.WL[:-5], self.N_5M_AU[:-5], label= str(self.Temp) + " C")
        plt.xlim((350,800))
        plt.xlabel('Wavelength (nm)', fontsize=25)
        plt.ylabel('Normalized Absorbance', fontsize =25)
        plt.ylim((0,1))
        plt.xlim((self.WL[0], self.WL[-5]))
        plt.tick_params( labelsize = 20, width=2, length=7)
        plt.legend(loc='upper right', frameon=False, fontsize=15)
        return


def Extract_Data(File_List, Range):
    # Function for extracting data
    N = len(File_List)
    AU_List = []
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
                    #row = row[0].split('\t')
                    WL_Temp = int(row[0])
                    AU_Temp = float(row[1])
                    #print WL_Temp, AU_Temp
                    WL.append(WL_Temp)
                    AU.append(AU_Temp)
                except:
                    continue
        #print AU, WL
        #print "Fuck you"
        AU = np.asarray(AU[Range[0]:Range[1]])
        WL = np.asarray(WL[Range[0]:Range[1]])
        A = Absorbance(Name, Temp, AU, WL)
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


def Fit_Lines(T,D):
    T = np.asarray(T)
    D = np.asarray(D)
    N = len(T)
    res = []
    res_array = np.empty(1)
    m = 0
    for i in range(3, N-3):
        m1, b1, r1, p1, std1= stats.linregress(T[0:i], D[0:i])
        m2, b2, r2, p2, std2 = stats.linregress(T[i+m:N], D[i+m:N])
    
        Tg = (b1 - b2)/(m2 -m1)
    
        res.append(i*r1**2 + (N-i)*r2**2)
        print res[-1]
        plt.plot(T, D, linestyle = '-', marker='o', color='k')
        D = np.asarray(D)
        plt.ylim((D.min()-0.01, D.max() + 0.01))
        plt.plot(T, m1*T + b1, 'r', label = "%d = %.2f" % (Tg, res[-1]))
        plt.plot(T, m2*T + b2, 'r')
    
        print Tg
        plt.xlabel("Annealing Temperature (C)", fontsize=25)
        plt.ylabel("Deviation Metric", fontsize=25)
        plt.legend()

    
    #plt.show()
    res_array = np.append( res_array, np.asarray(res))
    res = []
    #plt.plot(res)
    #plt.ylabel( "Sum of R$^2$")
    #plt.xlabel( "Increment")
    #plt.show()
    
    Inc =  np.argmax(res_array)+2
    print res
    print Inc
    m1, b1, r1, p1, std1= stats.linregress(T[0:Inc], D[0:Inc])
    m2, b2, r2, p2, std2 = stats.linregress(T[Inc+m:N], D[Inc+m:N])
    Tg = (b1 - b2)/(m2 -m1)
    plt.plot(T, m1*T + b1, 'g', linewidth=3)
    plt.plot(T, m2*T + b2, 'g', linewidth=3, label = '%d' % Tg)
    plt.legend(loc='upper left', frameon=False, fontsize=25)
    plt.tick_params( labelsize = 20, width=2, length=7)
    print Tg
    plt.show()
    return m1, b1, m2, b2, Tg


def Plot_SQD(AU_List):
    if AU_List[0].As_Cast:
        print "Found As Cast"
    else:
        print "No Ass Cast? WTF? Self Destruct"
        return
    MSD = []
    Temp = []
    for i in range(1,len(AU_List)):
        MSD_Temp = np.sqrt(np.sum((AU_List[0].N_AU - AU_List[i].N_AU)**2))
        MSD_Temp = MSD_Temp**2
        MSD.append(MSD_Temp)
        Temp.append(AU_List[i].Temp)
    Temp = np.asarray(Temp)
    MSD = np.asarray(MSD)

    m1, b1, m2, b2, Tg = Fit_Lines(Temp,MSD)
    plt.plot(Temp, m1*Temp + b1, 'r', linewidth=5)
    plt.plot(Temp, m2*Temp + b2, 'r', linewidth=5)
    plt.plot(Temp, MSD, 'o', markersize=10, color='k')
    plt.ylim((0, max(MSD)))
    plt.tick_params( labelsize = 20, width=2, length=7)
    plt.ylabel('Deviation Metric', fontsize=25)
    plt.xlabel('Temperature (C)', fontsize=25)
    plt.legend(loc='upper right', frameon=False, fontsize=25)
    plt.axvline(x=Tg,linewidth=4, color='g', alpha=0.5);
    plt.show()
    return






# Commands to be executed
Name = "P3BT"
Range = [160,610]

File_List = glob.glob("%s_*.CSV" % Name )
X= []

for File in File_List:
    X.append(int(File.split("_")[1]))
    Index = File.split("_")[2]

X = sorted(X)
File_List2 = []

for Value in X:
    File_List2.append("%s_%d_%s" % (Name, Value, Index))


from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
rcParams['axes.color_cycle']= sns.color_palette('coolwarm', len(File_List2))

                      
AU_List = Extract_Data(File_List2, Range)
Plot_UV(AU_List)
Plot_SQD(AU_List)

                       


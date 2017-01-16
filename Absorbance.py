#Python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks', {'xtick.direction':u'in', 'ytick.direction':u'in', 'image.cmap': u'coolwarm'})


class Absorbance():
    # Class for storing UV-VIs Data's
    def __init__(self, Name, Temp, AU, WL):
        self.Name = Name #String
        self.Temp_P = Temp #Int
        self.Temp_F = float(self.Temp_P)*0.9833 + 0.7451
        self.AU = AU # Numpy array of floats
        self.WL = WL # Numpy array of ints
        self.Ev = 1239.8/self.WL
        self.As_Cast = False
        # Put all your default parameters
        return
    
    def Print_Info(self):
        # Prints the name and the temperature
        print self.Name, self.Temp_F
        return
    
    def Smooth(self):
        # Performs a 5-point smoothing operation
        self.AU_5M = []
        self.WL_5M = []
        for i in range(2,len(self.AU)-2):
            self.AU_5M.append((self.AU[i-2] + self.AU[i-1] + self.AU[i]+ self.AU[i+1] + self.AU[i+2])/5.0)
            self.WL_5M.append(self.WL[i])
        self.AU_5M = np.asarray(self.AU_5M)
        self.WL_5M = np.asarray(self.WL_5M)
        return
    
    def Normalize(self):
        # Normalize the data
        self.N_AU = self.AU/max(self.AU)
        self.N_AU_5M = self.AU_5M/max(self.AU_5M)
    
    def Plot_Info(self):
        plt.plot(self.WL_5M, self.N_AU_5M, label = '%d' % self.Temp_F + " $^\circ$C")
        plt.xlabel('Wavelength (nm)', fontsize=25)
        plt.ylabel('Normalized Absorbance', fontsize=25)
        plt.ylim((0,1))
        plt.tick_params(labelsize=20, width=2, length=7)
        plt.legend(loc="upper right", frameon=False, fontsize=15)
        #plt.show()
        return

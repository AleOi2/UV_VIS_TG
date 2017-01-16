#!usr/bin/python

from Absorbance import *
import Aux_Functions as Aux
import numpy
import matplotlib.pyplot

def main():
    # Polymers and associated index/wavelength ranges:
    # P3BT [110-610]/[300-800 nm]
    # P3BT-PCBM
    # PDTSTPD
    # F8BT
    # PBTTT-C14
    # MEHPPV
    # PTB7
    Name = 'P3BT'
    Range = [110,610]

    Sorted_Files = Aux.Prepare_File_List( Name)
    AU_List = Aux.Extract_Data( Sorted_Files, Range)
    Aux.Plot_UV(AU_List)
    Aux.Plot_DM(AU_List)
    return



if __name__=='__main__': main()

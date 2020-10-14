import sys
sys.path.append("/home/vasileios.skliris/mly/")
from mly.datatools import DataPod
from mly.tools import *
import pylab
import os

import random
import numpy as np
import matplotlib.pyplot as plt

from pycbc.waveform import get_td_waveform
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries


def projectWave(sourceWaveform
                ,detectors
                ,fs
                ,declination=None
                ,rightAscension=None
                ,polarisationAngle=None
                ,time=10000000
                ,padCrop=1
                ,outputFormat=None
                ,destinationFile=None
                ,saveName=None):
    
    # # # PARAMETER CHECKS
    
    extraPlugins={}
    # Case where input is normaly a tuple or list o numpy array with the two polarisations 
    if isinstance(sourceWaveform,(list,tuple,np.ndarray)) and len(sourceWaveform)==2:
        h=sourceWaveform
        if outputFormat == None: 
            outputFormat='txt'

    # Case where input is a path to a file
    elif isinstance(sourceWaveform,str):
        # Case where source file is a txt file
        if sourceWaveform[-4:]=='.txt':
            h=np.loadtxt(sourceWaveform)
            if outputFormat == None: outputFormat='txt'
        # Case where source file is a DataPod (pickle) file
        elif sourceWaveform[-4:]=='.pkl':
            h=DataPod.load(sourceWaveform).strain
            if outputFormat == None: 
                outputFormat='DataPod'
                extraPlugins=sourceWaveform.pluginDict
        else:
            raise TypeError("File format is not supported")
            
    # Making the polarisation data into TimeSeries objects
    hp=TimeSeries(h[0],delta_t=1./fs,epoch=time)         
    hc=TimeSeries(h[1],delta_t=1./fs,epoch=time)      
    
    # Checking the detectors input
    if not all(d in ['H1','L1','V1'] for d in detectors):
        if all(d+'1' in ['H1','L1','V1'] for d in detectors):
            detectors = list(d+'1' for d in detectors)
        else:
            raise ValueError("Detectors can be only of H1, L1, V1")
    
    # Checking the rightAscension detectors input
    if rightAscension==None:
        rightAscension = 2*np.pi*np.random.rand()
    if not (0<= rightAscension <= 2*np.pi):
        raise ValueError("Right Ascension must be in [0,π]")  
        
    # Checking the declination input
    if declination==None:
        # In the case of declination we need a different
        # distribution to so that it is uniformal on the sky
        res=np.pi/360
        ph=np.arange(-np.pi/2,np.pi/2,res)
        prob=(res*np.cos(ph)/2)
        prob=prob/sum(prob)
        declination =np.random.choice(ph,p=prob)
    if not (-np.pi/2<= declination <= np.pi/2):
        raise ValueError("Declination must be in [-π/2,π/2]")
        
    # Checking the polarisationAngle input
    if polarisationAngle==None:
        polarisationAngle = 2*np.pi*np.random.rand()
    if not (0<= polarisationAngle <= 2*np.pi):
        raise ValueError("Polarisation Angle must be in [0,π]")
    polarisationAngle = 2*np.pi*np.random.rand()
    
    
    # # # PROJECTION - TIMESHIFT
    
    detector_dict={} # Dictionary with detector objects
    shift_dict={}    # Dictionary with shift for each detector (pixels)
    signal_dict={}   # The final projected signals in dictionary form
    
    for det in detectors:
        detector_dict[det]=Detector(det)
        # Calculation of time delay using as a reference the first detector
        dt = detector_dict[det].time_delay_from_detector(detector_dict[detectors[0]]
                                                         ,rightAscension
                                                         ,declination
                                                         ,time)
        # Changing the time delaY to pixel delay
        shift_dict[det]=int(dt*fs) 
        # Projecting the waveforms to the detectors
        signal = np.array(detector_dict[det].project_wave(hp, hc
                                                         ,rightAscension
                                                         ,declination
                                                         ,polarisationAngle))
        signal_dict[det]= np.array(signal)
        
        
    # There is a one pixel inconsistency sometimes and we make sure all signals have the same length
    if not all(len(signal_dict[det])==len(signal_dict[detectors[0]]) for det in detectors):
        maxlen=max(list(len(signal_dict[det])for det in detectors))
        if len(signal_dict[det])<maxlen: 
            signal_dict[det]=np.hstack((signal_dict[det],np.zeros(maxlen-len(signal_dict[det]))))
              
    # Shifting the signals acording to shift_dict
    for det in detectors:    
        signal_dict[det]=np.roll(signal_dict[det],shift_dict[det])[padCrop:-padCrop]
        # padCrop: Due to shifting the project_wave function pads with zeros to avoid
        # edge effects. Eventually we might want to crop after we finish. Avoid if not necessary 
    
    # # # OUTPUT FORMAT - SAVING 
    
    outputDict={}
    if outputFormat=='txt':
        for key in signal_dict.keys(): outputDict[key[0]]=signal_dict[key]
    elif outputFormat in ['DataPod']:
        for key in signal_dict.keys():
            # Creating the dataPod for each detector
            pod=DataPod([signal_dict[key]],detectors=key[0],fs=fs)
            # Adding any plugin info from the source file is any
            for pl in extraPlugins.values() : pod.addPlugIn(pl)
            outputDict[key[0]]=pod

    if saveName == None:
        if not isinstance(sourceWaveform,str):
            name='projectedGW'
        else:
            name=sourceWaveform.split('/')[-1].split('.')[0]
           
    if destinationFile == None:
        return(outputDict)
    elif isinstance(destinationFile,str):
        for det in detectors:
            if outputFormat=='txt':
                np.savetxt(destinationFile+det[0]+'/'+name+'.txt',outputDict[det[0]])
            elif outputFormat in ['DataPod']:
                outputDict[det[0]].save(destinationFile+det[0]+'/'+name+'.pkl')
                

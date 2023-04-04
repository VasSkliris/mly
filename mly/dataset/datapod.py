from .core import DataPodBase

from ..tools import dirlist, internalLags, correlate
from ..plugins import *
from ..simulateddetectornoise import *  
from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment,SegmentList,DataQualityFlag
from dqsegdb2.query import query_segments
from gwpy.io.kerberos import kinit
import gwdatafind


from gwpy.time import to_gps
from gwpy.time import from_gps
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from scipy.stats import pearsonr
from scipy.special import comb
from pycondor import Job, Dagman
from urllib.error import HTTPError

import numpy as npl
import pickle
import os
import sys
import time
import random
import copy
from math import ceil

################################################################################

class DataPod(DataPodBase):

    '''DataPod is an object to consentrate all the information of a data.
    instance. The reason for their creation is to encapsulate all the usefull
    attributes a machine learning algorithm will need. Getting them together 
    into DataSet objects creates a powerfull tool to simplify machine learning
    training and testing procedures for any kind of network.
    
    Attributes
    ----------
    
    strain :  numpy.ndarray / gwpy.timeseries.TimeSeries / list
        The main data of the pod. It can be a timeseries or a picture or 
        anything with fixed dimentions. The input is checked for nan
        and inf values and raises error if the data have size or value problems.
            
    fs : int
        This is the sample frequency of the data. It is used to determine 
        the duration of the of the strain input. If the strain is not a
        time series it is not used anywhere.
            
    labels : dict (optional)
        It determines the main labels of the DataPod. These labels will be
        used by algorithms for classification. If empty, a label called 
        "UNDEFINED" is given. Suggested keys for labels are the following:
        
              { 'type' : ('noise' , 'cbc' , 'signal' , 'burst' , 'glitch', ...),
                'snr'  : ( any int or float number bigger than zero),
                'delta': ( Declination for sky localisation, float [-pi/2,pi/2])
                'ra'   : ( Right ascention for sky licalisation float [0,2pi) )}
                
                For all labels, keys and values are not case sensitive.
            
    detectors : list (optional)
        If the strain belongs to a specific detector it is suggested to 
        state which detector is it. If it not specified the number of 
        detectors is assumed to be the smallest strain dimention and a
        list of 'U' is set with length equal to the detector number that 
        was assumed. The first dimention of every strain must be the 
        number of detectors involved. If it's not a check takes place to
        see if the dimention is somewhere else. If it is we the strain is
        transposed.
              
    gps : int / float (optional)
        The gps time of each of the detector strains. If not defined a
        list of zeros with length equal to detectors is defined. It has to
        be a possitive number.
    
    duration : int / float (optional for timeseries)
        The duration in seconds of the strain. If the strain is more than
        2D in shape the duration has to be difined independently.
              
    metadata : dict (optional)
        A dictionary with additional infromation about the pod.
        
    plugins : Plugin / str (optional)
        Except strain data it might be wanted to have other related data. 
        You can define PlugIn objects and pass them into the Dataset. Some
        of those PlugIns like 'correlation', are already defined and you can 
        call them as by passing their names as strings.
    '''
        
    def __getitem__(self,index):
        if isinstance(index,int): index=slice(index,None,None)
        if isinstance(index,slice):
            return self._strain[index]
        elif isinstance(index,list) and all((isinstance(det,str) and (det in self._detectors) 
             and det!='U') for det in index):
            newStrain=[]
            newGPS=[]
            newMeta=copy.deepcopy(self.metadata)
            ignoredDetectors=[]
            for d in self.detectors:
                if not (d in index) : ignoredDetectors.append(d)
            for det in index:
                ind = self._detectors.index(det)
                newStrain.append(self._strain[ind])
                newGPS.append(self.gps[ind])
            for d in ignoredDetectors:
                for key in newMeta:
                    if isinstance(newMeta[key],dict) and len(self.metadata[key])==len(self.detectors):
                        del newMeta[key][d]
            return(DataPod(newStrain,fs=self.fs,labels=copy.deepcopy(self.labels)
                           ,gps = newGPS,detectors=index
                           ,duration=self.duration
                           ,metadata=newMeta))
        else:
            raise TypeError("index can be int or a list with detector names that exists")
    
      
    # --- saving method ------------------------------------------------------ #
    
    def save(self, name=None
             , saving_format = None
             , allowed_formats = ['pkl','txt']):
        """Method to save the DataPod object

        Parameters
        -------------
        name : str 
            Name of the file to be saved.

        type_ : '.pkl' or .'txt'
            The format to save it.

        """
        if name is None:
            raise TypeError("You need to provide a name for the saved file.")

        elif isinstance(name,str):

            if any("."+f in name for f in allowed_formats):
                name = '.'.join(name.split(".")[:-1])

        if saving_format is None:
            saving_format = 'pkl'

        if saving_format == 'pkl' and (saving_format in allowed_formats):
            with open(finalName+'.pkl', 'wb') as output:
                pickle.dump(self, output, 4)

        elif saving_format == 'txt' and (saving_format in allowed_formats):
            np.savetxt(finalName+'.txt', self.strain)

        else:
            raise TypeError(saving_format+" is not supported.")
            
    # --- loading method ------------------------------------------------------#    
    
    def load(filename):
        """Method to load a DataPod object
        
        Parameters
        -------------
        filename : str (path)
            Path to the file.
        
        """
        if ".pkl" in filename:
            with open(filename,'rb') as obj:
                a = pickle.load(obj)
            return(a)
        elif ".txt" in filename:
            nArray = np.loadtxt(filename)
            return(nArray)
        else:
            raise TypeError("Only .pkl files are supported for loading")

    
    
    def addPlugIn(self,*args):
        """Method to add PlugIn objects to the dataPod. PlugIn objects are extra
        data derived either from the already existing data in the pod or tottaly
        new ones. You can add a simple value to generation function and also a
        plot function if you need to plot the new data.

        Parameters
        ----------

        *args: PlugIn objects
            The PlugIn objects that you have already defined and you want to add
            on the pod.

        """
        for plugin in args:
            if not isinstance(plugin,PlugIn):
                raise TypeError("Inputs must be PlugIn objects")
                
        for plugin in args:
            plugAttributes=[]
            for at in plugin.attributes:
                if at in dir(self):
                    plugAttributes.append(self.__getattribute__(at))
                else:
                    raise AttributeError(at+" is not part of DataPod instance")
            if callable(plugin.genFunction):
                self.__setattr__(plugin.name,plugin.genFunction(*plugAttributes,**plugin.kwargs))
            else:
                self.__setattr__(plugin.name,plugin.genFunction)

            self.pluginDict[plugin.name]=plugin
            
            
    def plot(self,type_='strain'):
        
        """A visualisation function for the DataPod. It will plot the data 
        currently present on the DataPod object, following LIGO colour 
        conventions.
        
        Parameters
        ----------
        
        type_ : str (optional)
            The type of data to plot. Some pods might have plottable PlugIn data.
            In this case you can provide the name of that PlugIn to be ploted. 
            Currently only strain, psd and correlation are supported. The
            default value is strain.
        
        """
        
        colors = {'H': '#ee0000','L':'#4ba6ff','V':'#9b59b6','K':'#ffb200','I':'#b0dd8b','U':'black'}
        names=[]
        if type_ == 'strain':
            
            plt.figure(figsize=(15,7))#,facecolor='lightslategray')
            if 'U' in self.detectors:
                names = list('Strain '+str(i) for i in range(len(self.detectors)))
            else:
                names = list(det for det in self.detectors)

            plt.xlabel('Time')
            plt.ylabel('Strain')
            t=np.arange(0,self.duration,1/self.fs)

            minim=0
            for i in range(len(self.detectors)):
                plt.plot(t,self.strain[i]+minim,color= colors[self.detectors[i]],label = names[i] )
                minim =max(self.strain[i]+minim+abs(min(self.strain[i])))
            plt.legend()
            
        elif type_ in dir(self):
            
            plugin=self.pluginDict[type_]
            plugAttributes=[]
            for at in plugin.plotAttributes:
                plugAttributes.append(self.__getattribute__(at))
            self.pluginDict[type_].plotFunction(*plugAttributes,data=self.__getattribute__(type_))
        
        else:
            raise ValueError("The type specified is not present in the DataPod or it does not have a plot function.")


            
            
            

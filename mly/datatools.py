from .core import DataPodBase
from .core import DataSetBase
from .tools import dirlist, internalLags, correlate
from .plugins import *
from .simulateddetectornoise import *  
from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment,SegmentList,DataQualityFlag
from dqsegdb2.query import query_segments
from gwpy.io.kerberos import kinit

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
#  TODO:You have to make nan checks to the injection files and noise files 
#  before running the generator function.
# 
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
        anything with fixed dimentions. Finally it checks the input for nans
        and infs and returns errors if the data have size or value problems.
            
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
    
    def save(self, name='',type_ = 'pkl'):
        """Method to save the DataPod object
        
        Parameters
        -------------
        name : str 
            Name of the file to be saved.
        
        type_ : '.pkl' or .'txt'
            The format to save it.
        
        """
        if name == '':
            finalName = 'dataPodNameToken'+str("%04d" %
                (np.random.randint(0,10000)))+'.'+type_
        elif ('.pkl' == name[-4:]) or ('.txt' == name[-4:]):
            finalName = name[:-4]
        else:
            finalName = name
            
        if type_ == 'pkl':
            with open(finalName+'.pkl', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        elif type_ == 'txt':
            np.savetxt(finalName+'.txt', self.strain)
        else:
            raise TypeError("."+type_+" files are not supported for saving")
            
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
        data derived either from the already existing data in the pod or tottaly new
        ones. You can add a simple value to generation function and also a plot function
        if you need to plot the new data.

        Parameters
        ----------

        *args: PlugIn objects
            The PlugIn objects that you have already defined and you want to add on the pod.

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
            self.__setattr__(plugin.name,plugin.genFunction(*plugAttributes,**plugin.kwargs))

            self.pluginDict[plugin.name]=plugin
            
            
    def plot(self,type_='strain'):
        
        """A visualisation function for the DataPod. It will plot the data currently present
        on the DataPod object, following LIGO colour conventions.
        
        Parameters
        ------------
        
        type_ : str (optional)
            The type of data to plot. Some pods might have plottable PlugIn data. In this case
            you can provide the name of that PlugIn to be ploted. Currently only strain, psd and
            correlation are supported. The default value is strain.
        
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

              
class DataSet(DataSetBase):
    
    """DataSet is an object that helps manipulate groups of DataPods as a whole.
    The main attribute is a list of the DataPods. All methods are providing ways
    to manipulate the data and export them to desired shapes. Finally it provides
    a DataSet generator.
    
    Attributes
    ----------
    
    dataPods: list of DataPod objects (optional)
        List of the DataPod objects to be part of this DataSet instance. All DataPods
        are checked for inconsistences such as different shapes, different number of
        detectors and different sample frequencies.
    
    name: str (optional)
        The name of the DataSet
    
    """
    
    
    def __getitem__(self,index):
        if isinstance(index,int):
            return self._dataPods[index]
        elif isinstance(index,slice):
            return DataSet(self._dataPods[index])
        elif (isinstance(index,list) 
              and all(isinstance(det,str) for det in index)):
            newPods=[]
            for pod in self._dataPods:
                newPods.append(pod[index])
            return(DataSet(newPods))
                
    
    # --- saving method -------------------------------------------------------#
    
    def save(self, name = None,type_ = 'pkl'):
        
        """Method to save the DataSet.
        
        Parameters
        -------------
        name : str  (optional)
            Name of the file to be saved. If not specified the name becomes the 
            name attribute.
        
        type_ : '.pkl'
            The format to save the DataSet. Currently available only as pkl.
        
        """       
        if name == None : name= self.name
        if name == None:
            finalName = 'dataSetNameToken'+str("%04d" %
                (np.random.randint(0,10000)))+'.'+type_
        else:
            finalName = (name+'.'+type_)
        if type_ == 'pkl':
            with open(finalName, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        else:
            raise TypeError("."+type_+" files are not supported for saving")
            
    # --- loading method ------------------------------------------------------#    
    
    # needs chop
    def load(filename, source = 'file'):
        
        """Method to load a DataSet object
        
        Parameters
        -------------
        filename : str (path)
            Path to the file.
        
        """
        
        if source == 'file':
            if ".pkl" in filename:
                with open(filename,'rb') as obj:
                    set_ = pickle.load(obj)
                if isinstance(set_,DataSet):
                    return(set_)
                else:
                    raise TypeError("Object in file is not a DataSet object")
            else:
                raise TypeError("Only .pkl files are supported for loading")
                
        elif source == 'directory':
            
            det=dirlist(filename)
            if filename[-1] == '/' : filename.pop[-1]
            detectors=[]
            for i in range(len(det)):
                if det[i] in ['H','L','V','K','I',]:
                    detectors.append(det[i])
            if len(detectors)==0: 
                raise ValueError("Directory does't have detector data")
                
            filelist=[]
            for det in detectors:
                filelist.append(dirlist(filename +'/'+ det))

            filelist = np.asarray(filelist)
            index=np.arange(len(filelist[0]))
            np.random.shuffle(index)
            for i in range(len(filelist)):
                filelist[i] = filelist[i,index]

            pods=[]      
            for i in range(len(filelist[0])):

                data=[]  
                for det in detectors:
                    data.append( np.loadtxt(filename+'/'+det+'/'+filelist[detectors.index(det)][i]).tolist())
                pod = DataPod( data 
                               , fs =2048 , labels={'type': 'injection'}
                               , detectors = ['H','L','V']
                               , gps =None
                               , metadata = {'source file':filename})
                pods.append(pod)

            set_=DataSet(pods)
            return set_
        else:
            raise ValueError("source can be a 'file' or a 'directory'")
            
    # --- add method ---------------------------------------------------------#    
    
    def add(self, newData):
        """This method works similarly to how append works in lists. You can
        add a new DataPod or a new DataSet.
            
        
        
        """
        if isinstance(newData,DataPod):
            if self._dataPods==[]:
                pod0 = newData
            else:
                pod0=self._dataPods[0]
            if newData.shape != pod0.shape:
                print("Pods with different shapes")
            elif newData.fs != pod0.fs:
                print("Pods woth different sample frequencies")
            else:
                self._dataPods.append(newData)
        elif isinstance(newData, DataSet):
            pod0=self._dataPods[0]
            for pod in newData:
                if pod.shape != pod0.shape:
                    print("Pods with different shapes")
                if pod.fs != pod0.fs:
                    print("Pods woth different sample frequencies")
                if all(d in pod0.detectors for d in pod.detectors):
                    print("Pods with different detectors")
                else:
                    self._dataPods.append(newData.dataPods)
        else:
            raise TypeError("Appended object is not a DataPod or Dataset")
            
    def addPlugIn(self,*args):
        """Method to add PlugIn objects to the dataPod. PlugIn objects are extra
        data derived either from the already existing data in the pod or tottaly new
        ones. You can add a simple value to generation function and also a plot function
        if you need to plot the new data.

        Parameters
        ----------

        *args: PlugIn objects
            The PlugIn objects that you have already defined and you want to add on the pod.

        """
        for plugin in args:
            if not isinstance(plugin,PlugIn):
                raise TypeError("Inputs must be PlugIn objects")
                
        for plugin in args:
            plugAttributes=[]
            for at in plugin.attributes:
                if at in dir(self.dataPods[0]):
                    for pod in self.dataPods:
                        pod.addPlugIn(plugin)
                else:
                    raise AttributeError(at+" is not part of DataPod instance")

            self.pluginDict[plugin.name]=plugin
            
           
    # --- fusion method --        
    def fusion(elements, sizes = None, save = None):
        
        if not isinstance(elements,list): raise TypeError(
            "The fusion input must be a list containing either DataSet "+
            "object or file names, you can also have both")
        
        for element in elements:
            if isinstance(element, str):
                if not (os.path.isfile(element)):
                    raise FileNotFoundError("File "+element+" not found")
                elif not (".pkl" in element):
                    raise TypeError("File must be of the .pkl type")
            elif isinstance(element, DataSet):
                pass
            else:
                raise TypeError("The fusion input must be a list "
                    +"containing either DataSet object or file names,"
                    +" you can also have both")
        
        if isinstance(save, str):
            pass
        elif save == None:
            pass
        else:
            raise TypeError("save value can be a str or None")
                
        if sizes == None:
            pass
        else:
            if isinstance(sizes,list):
                if all(isinstance(x,int) for x in sizes):
                    if len(sizes) == len(elements):
                        pass
                    else:
                        raise TypeError("sizes must be equal in number "
                                        +"to the DataSets to fuse")
                else:
                    raise TypeError("sizes must be a list of integers or None")
            else:
                raise TypeError("sizes must be a list of integers or None")
        
        datasets=[]
        for i in range(len(elements)):
            if isinstance(elements[i], str):
                set_=DataSet.load(elements[i])
            elif isinstance(elements[i], DataSet):
                set_=elements[i]
            else:
                raise TypeError("element is not what expected")
                
            if sizes == None:
                datasets += set_._dataPods
            else:
                datasets += set_._dataPods[:sizes[i]]
        random.shuffle(datasets)
        finalSet = DataSet(datasets)
        
        if isinstance(save,str):
            finalSet.save(name= save)        
        
        return(finalSet)

    # --- filterGPS method --------------------------------------------------- #
    #                                                                          #
    # This method clears the DataSet from DataPods that include specific GPS   #
    # times. This can be done selectively for specific ditectors. The times    #
    # and intervals stated in filterTimes parameter are rejected for the       #
    # detectors stated in detectors parameter. The window parameter adds extra #
    # opotional discrimination at the intervals to be rejected.                #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    def filterGPS(self, filterTimes, detectors = None, window=0):
        
        if isinstance(filterTimes, (int,float)):
            fitlertTimes = [filterTimes]
        elif (isinstance(filterTimes, list) and all(
            isinstance(x,(int,float,list)) for x in filterTimes)):
            for x in filterTimes:
                if isinstance(x,list):
                    if not (len(x)==2 and all(
                        isinstance(t,(int,float)) for t in x) and x[1]>x[0]):
                         raise TypeError(
                             "filterTimes must be a list including gps times"
                            +" or intervals of gps times in the form of lists"
                             +" [start,end)")
        else:
             raise TypeError(
                 "filterTimes must be a list including gps times"
                +" or intervals of gps times in the form of lists [start,end)")
        
        if detectors != None:
            if isinstance(detectors,str): detectors = [detectors]
            if not (isinstance(detectors,list) 
                    and all(d in ['H','L','V','K','I','U'] for d in detectors)):
                raise TypeError(
                "detectors have to be a list of strings"+
                " with at least one the followings as elements: \n"+
                "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
                "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India"+
                " (INDIGO) \n\n'U' if you don't want to specify detector")

        
        rejectionInterval=SegmentList()
        for tm in filterTimes:
            if not isinstance(tm,list):
                rejectionInterval.append(Segment(tm-0.5,tm+0.5))
            else:
                rejectionInterval.append(Segment(tm[0],tm[1]))
        rejectionInterval.coalesce()
        filteredPods=[]
        for i in range(len(self.dataPods)):
            exlusion = False
            
            if detectors==None:
                for j in range(len(self.dataPods[i].detectors)):
                    if rejectionInterval.intersects_segment(Segment(
                        self.dataPods[i].gps[j]-window,
                        self.dataPods[i].gps[j]+
                        self.dataPods[i].duration+window)):
                        exlusion = True
    
            else:
                for d in detectors:
                    ind = self.dataPods[i].detectors.index(d)
                    if rejectionInterval.intersects_segment(Segment(
                        self.dataPods[i].gps[ind]-window,
                        self.dataPods[i].gps[ind]+
                        self.dataPods[i].duration+window)):
                        exlusion = True
            
            if exlusion == False:
                filteredPods.append(self.dataPods[i])
                
        self._dataPods=filteredPods
        print("Given GPS times, no longer in DataSet")
    
    def filterDetector(self,detectors):
        
        if isinstance(detectors,str): detectors = [detectors]
        if not (isinstance(detectors,list) 
                and all(d in ['H','L','V','K','I','U'] for d in detectors)):
            raise TypeError(
            "detectors have to be a list of strings"+
            " with at least one the followings as elements: \n"+
            "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
            "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
            "\n'U' if you don't want to specify detector")
            
        for i in range(len(self.dataPods)):
            for d in detectors:
                ind = self.dataPods[i].detectors.index(d)                
                self._dataPods[i]._strain = np.delete(self._dataPods[i]._strain
                                                      , ind , 0)
                self._dataPods[i]._detectors = np.delete(
                    self._dataPods[i]._detectors, ind, 0)
                self._dataPods[i]._gps = np.delete(self._dataPods[i]._gps
                                                   , ind, 0)
        message = "Detector"
        if len(detectors)==1:
            message+=" "
        else:
            message+="s "
        for d in detectors:
            message+=(d+", ")
        message += "no longer in this DataSet"
        print(message)
        
    def filterLabel(self,labels):
        
        if not (isinstance(labels, dict) and len(labels)>0):
            raise ValueError("labels have to be a dict with at least"
                             +" one key-value pare")
        
        finalPods=[]
        for i in range(len(self._dataPods)):
            
            same = 0
            for j in range(len(self._dataPods[i].labels)):
                for k in range(len(labels)):
                    if ((list(labels.keys(
                    ))[k]==list(self._dataPods[i].labels.keys())[j])
                        and (labels[list(labels.keys(
                        ))[k]]==self._dataPods[i].labels[list(
                            self._dataPods[i].labels.keys())[j]])):
                        same+=1
            if same == len(labels):
                finalPods.append(self._dataPods[i])
                
        self._dataPods = finalPods
        

    def exportData(self,plugin=None, shape = None):

        goods =[]
        if plugin==None:
            for pod in self.dataPods:
                goods.append(pod.strain.tolist())
            goods=np.array(goods)                

        elif isinstance(plugin,str):
            for pod in self.dataPods:
                goods.append(pod.__getattribute__(plugin))
            goods=np.array(goods)
        else:
            raise ValueError('PlugIn '+str(extras)+' not present in DataPod')
            
        if shape == None:
            shape = goods.shape
        if isinstance(shape,tuple):

            if all(((dim in goods.shape) or dim==None) for dim in shape):
                shapeList = list(shape)
                print(shapeList)
                goodsShapeList = [None]+list(goods.shape)[1:]
                print(goodsShapeList)
                newIndex = list(shapeList.index(goodsShapeList[i]) for i in range(len(shape)))
                print(newIndex)
                goods = np.transpose(goods, newIndex)
            else:
                raise ValueError("Shape values are not the same as the DataSet shape")
#             if None in goods.shape: 
#                 shapeList=list(shape)
#                 shapeList[0]=len(self)
#                 shape=tuple(shapeList)
        else:
            raise TypeError("Not valid shape.")
        print("DataSet with shape "+str(goods.shape)+" is exported")
        return(goods)
    
    def exportLabels(self,*args,reshape=False):
        # Checking the types of labels and how many
        # pods have the specific label.
        labelOccur={}
        for pod in self.dataPods:
            for key in list(pod.labels.keys()):
                if key not in list(labelOccur.keys()): labelOccur[key]=0
                labelOccur[key]+=1
        # 'type' is a default label
        if len(args)==0: args=('type',)    
        for arg in args:
            # labels must be strings
            if not isinstance(arg,str):
                raise TypeError("Label keys must be strings")
            # if a label type doesn't exist an error is raised
            if not arg in list(labelOccur.keys()):
                raise KeyError("There is no label key "+str(arg))
            # The label type must occur as many times as the size of the dataset
            if labelOccur[arg]!=len(self):
                raise ValueError(str(len(self)-labelOccur[arg])
                                 +" pods don't have the label "+str(arg)+" defined")
                
        goods=[]
        for i in range(len(self)):
            if len(args)>1:
                label=[]
                for arg in args:
                    label.append(self[i].labels[arg])
            else:
                label=self[i].labels[arg]
            goods.append(label)
        goods=np.asarray(goods)
        # reshaping in case only one label is required
        if (len(args)==1 and reshape==True): goods=goods.reshape((len(goods),1))
        print("Labels "+str(list(args))+" with shape "+str(goods.shape)+" are unloaded")
        return goods
   
    def exportGPS(self):
        goods =[]
        for pod in self.dataPods:
            goods.append(pod.gps)
        return goods
        
    def generator(duration
                   ,fs
                   ,size
                   ,detectors
                   ,injectionFolder = None
                   ,labels = None
                   ,backgroundType = None
                   ,injectionSNR = None
                   ,noiseSourceFile = None  
                   ,windowSize = None #(32)            
                   ,timeSlides = None #(1)
                   ,startingPoint= None #(32)
                   ,name = None
                   ,savePath = None
                   ,single = False  # Making single detector injections as glitch
                   ,injectionCrop = 0  # Allows to crop part of the injection when you move the injection arroud, 0 is no 1 is maximum means 100% cropping allowed. The cropping will be a random displacement from zero to parto of duration
                
                   ,disposition=None
                   ,maxDuration=None
                   ,differentSignals=False   # In case we want to put different injection to every detector.
                   ,plugins=None):

        # Integration limits for the calculation of analytical SNR
        # These values are very important for the calculation

        fl, fm=20, int(fs/2)#

        profile = {'H' :'aligo','L':'aligo','V':'avirgo','K':'KAGRA_Early','I':'aligo'}

        # Labels used in saving file
        #lab={10:'X', 100:'C', 1000:'M', 10000:'XM',100000:'CM'}  

        lab={}
        if size not in lab:
            lab[size]=str(size)

        # ---------------------------------------------------------------------------------------- #   
        # --- duration --------------------------------------------------------------------------- #

        if not (isinstance(duration,(float,int)) and duration>0 ):
            raise ValueError('The duration value has to be a possitive float'
                +' or integer representing seconds.')

        # ---------------------------------------------------------------------------------------- #   
        # --- fs - sample frequency -------------------------------------------------------------- #

        if not (isinstance(fs,int) and fs>0):
            raise ValueError('Sample frequency has to be a positive integer.')

        # ---------------------------------------------------------------------------------------- #   
        # --- detectors -------------------------------------------------------------------------- #

        if isinstance(detectors,(str,list)):
            for d in detectors:
                if d not in ['H','L','V','K','I']: 
                    raise ValueError("detectors have to be a list of strings or a string"+
                " with at least one the followings as elements: \n"+
                "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
                "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
                "\n'U' if you don't want to specify detector")
            if isinstance(detectors,str): detectors=list(detectors)
        else:
            raise ValueError("detectors have to be a list of strings or a string"+
                " with at least one the followings as elements: \n"+
                "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
                "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
                "\n'U' if you don't want to specify detector")
        

        # ---------------------------------------------------------------------------------------- #   
        # --- size ------------------------------------------------------------------------------- #        

        if not (isinstance(size, int) and size > 0):
            raise ValueError("size must be a possitive integer.")

        # ---------------------------------------------------------------------------------------- #   
        # --- injectionFolder -------------------------------------------------------------------- #

        if injectionFolder == None:
            pass
        elif isinstance(injectionFolder, str):            
            if (('/' in injectionFolder) and os.path.isdir(injectionFolder)):
                injectionFolder_set = injectionFolder.split('/')[-1]
            elif (('/' not in injectionFolder) and any('injections' in p for p in sys.path)):
                    for p in sys.path:
                        if ('injections' in p):
                            injectionFolder_path = (p.split('injections')[0]+'injections'
                                +'/'+injectionFolder)
                            if os.path.isdir(injectionFolder_path):
                                injectionFolder = injectionFolder_path
                                injectionFolder_set = injectionFolder.split('/')[-1]
                            else:
                                raise FileNotFoundError('No such file or directory:'
                                                        +injectionFolder_path)

            else:
                raise FileNotFoundError('No such file or directory:'+injectionFolder) 
        else:
            raise TypeError("cbcFolder has to be a string indicating a folder "
                            +"in MLyWorkbench or a full path to a folder")

        # ---------------------------------------------------------------------------------------- #   
        # --- labels ----------------------------------------------------------------------------- #

        if labels == None:
            labels = {'type' : 'UNDEFINED'}
        elif not isinstance(labels,dict):
            raise TypeError(" Labels must be a dictionary.Suggested keys for labels"
                            +"are the following: \n{ 'type' : ('noise' , 'cbc' , 'signal'"
                            +" , 'burst' , 'glitch', ...),\n'snr'  : ( any int or float number "
                            +"bigger than zero),\n'delta': ( Declination for sky localisation,"
                            +" float [-pi/2,pi/2])\n'ra'   : ( Right ascention for sky "
                            +"localisation float [0,2pi) )})")

        # ---------------------------------------------------------------------------------------- #   
        # --- backgroundType --------------------------------------------------------------------- #

        if backgroundType == None:
            backgroundType = 'optimal'
        elif not (isinstance(backgroundType,str) 
              and (backgroundType in ['optimal','sudo_real','real'])):
            raise ValueError("backgroundType is a string that can take values : "
                            +"'optimal' | 'sudo_real' | 'real'.")

        # ---------------------------------------------------------------------------------------- #   
        # --- injectionSNR ----------------------------------------------------------------------- #
        if injectionFolder == None :
            injectionSNR = 0
        elif (injectionFolder != None and injectionSNR == None ):
            raise ValueError("If you want to use an injection for generation of"+
                             "data, you have to specify the SNR you want.")
        elif injectionFolder != None and (not (isinstance(injectionSNR,(int,float)) 
                                          and injectionSNR >= 0)):
            raise ValueError("injectionSNR has to be a positive number")


        # ---------------------------------------------------------------------------------------- #   
        # --- noiseSourceFile -------------------------------------------------------------------- #

        if (backgroundType == 'sudo_real' or backgroundType =='real'):

            if noiseSourceFile == None:
                raise TypeError('If you use sudo_real or real noise you need'
                    +' a real noise file as a source.')

            if (noiseSourceFile!=None and isinstance(noiseSourceFile,list) 
                    and len(noiseSourceFile)==2 
                    and all(isinstance(el,str) for el in noiseSourceFile)):

                if '.txt' in noiseSourceFile[1]:
                    noiseSourceFile[1] = noiseSourceFile[1][:-4]

                path_main=''
                path_check = False

                if (('/' in noiseSourceFile[0]) and all(os.path.isfile( noiseSourceFile[0]
                                +'/'+det+'/'+noiseSourceFile[1]+'.txt') for det in detectors)):
                    path_main = ''
                    path_check = True


                elif (('/' not in noiseSourceFile[0]) and any('ligo_data' in p for p in sys.path)):
                    for p in sys.path:
                        if ('ligo_data' in p):
                            path_main = (p.split('ligo_data')[0]+'ligo_data/'+str(int(fs))+'/')
                            if all(os.path.isfile(path_main+noiseSourceFile[0]+'/'+det+'/'
                                    +noiseSourceFile[1]+'.txt') for det in detectors):    
                                path_check = True
                                break
                            else:
                                raise FileNotFoundError("No such file or directory: "+path_main
                                                        +"/<detector>/"+noiseSourceFile[1]+".txt")
                else:
                    raise TypeError("Noise source file has to be a list of two strings:\n"
                                    +"--> The first is the path to the date folder that include\n "
                                    +"    the data of all the detectors or just the datefile given\n"
                                    +"    that the path is in sys.path.\n\n"
                                    +"--> The second is the file name of the segment to be used.")
                if path_check == False:
                    raise FileNotFoundError(
                        "No such file or directory: "+noiseSourceFile[0]
                        +"/<detector>/"+noiseSourceFile[1]+".txt")
            
            elif (noiseSourceFile!=None and isinstance(noiseSourceFile,list) 
                    and len(noiseSourceFile)==len(detectors) 
                    and all(len(el)==2 for el in noiseSourceFile)):
                pass
#                 seg_=Segment(noiseSourceFile[d][0], noiseSourceFile[d][1])
#                 print(seg_)
#                 # Open Data Channels
#                 det_seg_channels = {'H': 'H1_DATA','L': 'L1_DATA','V': 'V1_DATA'}
#                 # Feching the segment times that detectors were active
#                 for d in range(len(detectors)):
#                     print(noiseSourceFile[d])
#                     print(noiseSourceFile[d][0], noiseSourceFile[d][1])
                    

#                     try:
#                         seg_=DataQualityFlag.fetch_open_data(
#                             det_seg_channels[detectors[d]]
#                             , noiseSourceFile[d][0]
#                             , noiseSourceFile[d][1]).active
                        
#                     except:
#                         try:
#                             print('Trying non open data')
#                             det_seg_channels_ = {'H': 'H1:DMT-ANALYSIS_READY:1'
#                                                 ,'L': 'L1:DMT-ANALYSIS_READY:1'
#                                                 ,'V': 'V1:ITF_SCIENCE:1'}

#                             seg_= query_segments(
#                                 det_seg_channels_[detectors[d]]
#                                 , noiseSourceFile[d][0]
#                                 , noiseSourceFile[d][1])
                            
#                         except:
#                             try:
#                                 kinit(keytab= '/home/vasileios.skliris/vasileios.skliris.keytab')
#                                 os.system("ligo-proxy-init -k")
#                                 det_seg_channels_ = {'H': 'H1:DMT-ANALYSIS_READY:1'
#                                                     ,'L': 'L1:DMT-ANALYSIS_READY:1'
#                                                     ,'V': 'V1:ITF_SCIENCE:1'}

#                                 seg_= query_segments(
#                                     det_seg_channels_[detectors[d]]
#                                     , noiseSourceFile[d][0]
#                                     , noiseSourceFile[d][1])
#                             except:
#                                 print('everything failed')
#                                 raise


#                 print(seg_,len(seg_),seg_[0][1]-seg_[0][0],noiseSourceFile[d][1]-noiseSourceFile[d][0])
#                 if len(seg_)==0: 
#                     raise ValueError("No active segments during those gps times for "+detectors[d]+" detector.")
#                 elif len(seg_)>1:
#                     raise ValueError("There are more than one active segments during those gps times for "+detectors[d]+" detector.")
#                 elif (len(seg_)==1 and seg_[0][1]-seg_[0][0]!=noiseSourceFile[d][1]-noiseSourceFile[d][0]) : 
#                     raise ValueError("Only part of the required segment is active.")

        # ---------------------------------------------------------------------------------------- #   
        # --- windowSize --(for PSD)-------------------------------------------------------------- #        

        if windowSize == None: windowSize = duration*8
        if not isinstance(windowSize,int):
            raise ValueError('windowSize needs to be an integral')
        if windowSize < duration :
            raise ValueError('windowSize needs to be bigger than the duration')

        # ---------------------------------------------------------------------------------------- #   
        # --- timeSlides ------------------------------------------------------------------------- #

        if timeSlides == None: timeSlides = 0
        if not (isinstance(timeSlides, int) and timeSlides >=0) :
            raise ValueError('timeSlides has to be an integer equal or bigger than 0')

        # ---------------------------------------------------------------------------------------- #   
        # --- startingPoint ---------------------------------------------------------------------- #

        if startingPoint == None : startingPoint = 0 
        if not (isinstance(startingPoint, int) and startingPoint >=0) :
            raise ValueError('lags has to be an integer')        

        # ---------------------------------------------------------------------------------------- #   
        # --- name ------------------------------------------------------------------------------- #

        if name == None : name = ''
        if not isinstance(name,str): 
            raise ValueError('name optional value has to be a string')

        # ---------------------------------------------------------------------------------------- #   
        # --- savePath --------------------------------------------------------------------------- #

        if savePath == None : 
            pass
        elif (savePath,str): 
            if not os.path.isdir(savePath) : 
                raise FileNotFoundError('No such file or directory:' +savePath)
        else:
            raise TypeError("Destination Path has to be a string valid path")
            
        # ---------------------------------------------------------------------------------------- #   
        # --- plugins ---------------------------------------------------------------------------- #
        
        if plugins == None:
            plugins = []
        elif not isinstance(plugins,list):
            plugins = [plugins]
        if isinstance(plugins,list):
            for pl in plugins:
                print(pl,known_plug_ins)
                if pl in known_plug_ins:
                    pass
                elif isinstance(pl,PlugIn):
                    pass
                else:
                    raise TypeError("plugins must be a list of PlugIn object or from "+str(known_plug_ins))

            
        ### disposition
        # If you want no shiftings disposition will be zero
        if disposition == None: disposition=0
        # If you want shiftings disposition has to adjust the maximum length of an injection
        if isinstance(disposition,(int,float)) and 0<=disposition<duration:
            disposition_=disposition
            maxDuration_ = duration-disposition_
        # If you want random shiftings you can define a tuple or list of those random shiftings.
        # This will adjust also
        elif (isinstance(disposition,(list,tuple))):
            if not(len(disposition)==2 
                and isinstance(disposition[0],(int,float))
                and isinstance(disposition[1],(int,float)) 
                and 0<=disposition[0]<duration 
                and disposition[0]<disposition[1]<duration):
            
                raise ValueError('If you want random range of dispositons '
                                +'it needs to be a tuple or list of two numbers that represent a range')
            else: disposition_=disposition[1]
        else:
            raise TypeError('Disposition can be a number or a range'
                            +' of two nubers (start,end) that always is less than duration')
            
        # max Duration
        
        if maxDuration == None: 
            maxDuration=duration
        if not (isinstance(maxDuration,(int,float)) and 0<maxDuration<=duration):
            raise ValueError('maxDuration must be between 0 and duration')
        else:
            maxDuration_=min(duration-disposition_,maxDuration)
        
        # Making a list of the injection names,
        # so that we can sample randomly from them

        injectionFileDict={}
        noise_segDict={}
        for det in detectors:

            if injectionFolder == None:
                injectionFileDict[det] = None
            else:
                injectionFileDict[det] = dirlist(injectionFolder+'/' + det)

        if backgroundType == 'optimal':
            magic={1024: 2**(-21./16.), 2048: 2**(-23./16.), 4096: 2**(-25./16.), 8192: 2**(-27./16.)}
            param = magic[fs]

        elif backgroundType in ['sudo_real','real']:
            param = 1
            if isinstance(noiseSourceFile[0],str):
                for det in detectors:
                    noise_segDict[det] = np.loadtxt(path_main+noiseSourceFile[0]
                                                           +'/'+det+'/'+noiseSourceFile[1]+'.txt')    
                    gps0 = int(noiseSourceFile[1].split('_')[1])


            elif isinstance(noiseSourceFile[0],list):
                for d in range(len(detectors)):
                    
                    try:
                        det_channels_open = {'H1': 'H1_DATA'
                                            ,'L1': 'L1_DATA'
                                            ,'V1': 'V1_DATA'}
                        noise_segDict[detectors[d]] = TimeSeries.fetch_open_data(
                            det_channels_open[detectors[d]+'1']
                            ,noiseSourceFile[d][0]
                            ,noiseSourceFile[d][1]).astype('float64').resample(fs).value
                        
                    except:
                        try:
                            print('opened data failed ')
                            det_channels_get = {'H1': 'H1:GDS-CALIB_STRAIN'
                                               ,'L1': 'L1:GDS-CALIB_STRAIN'
                                               ,'V1': 'V1:Hrec_hoft_16384Hz'}

                            det_frametypes = {'H1': 'H1_HOFT_C00'
                                             ,'L1': 'L1_HOFT_C00'
                                             ,'V1': 'V1Online'}
                            noise_segDict[detectors[d]]=TimeSeries.get(
                                det_channels_get[detectors[d]+'1']
                                ,noiseSourceFile[d][0]
                                ,noiseSourceFile[d][1]
                                ,frametype=det_frametypes[detectors[d]+'1']
                                ,verbose=True).astype('float64').resample(fs).value
                            
#                         except RuntimeError:
#                             print('non public data get failed too')
#                             raise
                            
                        except:
                            try:
                                print('opened data failed again')
                                kinit(keytab= '/home/vasileios.skliris/vasileios.skliris.keytab')
                                #os.system("ligo-proxy-init -k")
                                
                                det_channels_get = {'H1': 'H1:GDS-CALIB_STRAIN'
                                                   ,'L1': 'L1:GDS-CALIB_STRAIN'
                                                   ,'V1': 'V1:Hrec_hoft_16384Hz'}

                                det_frametypes = {'H1': 'H1_HOFT_C00'
                                                 ,'L1': 'L1_HOFT_C00'
                                                 ,'V1': 'V1Online'}

                                noise_segDict[detectors[d]]=TimeSeries.get(
                                    det_channels_get[detectors[d]+'1']
                                    ,noiseSourceFile[d][0]
                                    ,noiseSourceFile[d][1]
                                    ,frametype=det_frametypes[detectors[d]+'1']
                                    ,verbose=True).astype('float64').resample(fs).value
                            except:
                                print('non public data get failed too')
                                raise
                            
                    print(len(noise_segDict[detectors[d]])/fs)
                    gps0 = int(noiseSourceFile[d][0])


                
            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs-windowSize+duration)
                               ,start_from_sec=startingPoint)
            #print(ind)



        thetime = time.time()
        DATA=DataSet(name = name)
        
        for I in range(size):
            print(I)
            detKeys = list(injectionFileDict.keys())
            
            if single == True: luckyDet = np.random.choice(detKeys)
            if isinstance(disposition,(list,tuple)):
                disposition_=disposition[0]+np.random.rand()*(disposition[1]-disposition[0])
                maxDuration_=min(duration-disposition_,maxDuration)
            
            index_selection={}
            if injectionFolder != None:
                if differentSignals==True:
                    if maxDuration_ != duration:
                        for det in detectors: 
                            index_sample=np.random.randint(0,
                                                      len(injectionFileDict[detKeys[0]]))
                            sampling_strain=np.loadtxt(injectionFolder+'/'
                                                       +det+'/'+injectionFileDict[det][index_sample])
                            while (len(sampling_strain)/fs > maxDuration_ 
                                   or index_sample in list(index_selection.values())):
                                index_sample=np.random.randint(0,
                                                          len(injectionFileDict[detKeys[0]]))
                                sampling_strain=np.loadtxt(injectionFolder+'/'
                                                           +det+'/'+injectionFileDict[det][index_sample])
                            
                            index_selection[det]=index_sample
                    else:
                        for det in detectors: 
                            index_sample=np.random.randint(0,
                                                      len(injectionFileDict[detKeys[0]]))
                            while (index_sample in list(index_selection.values())):
                                index_sample=np.random.randint(0,
                                                          len(injectionFileDict[detKeys[0]]))
                            index_selection[det]=index_sample
                else:
                    if maxDuration_ != duration:
                        index_sample=np.random.randint(0,len(injectionFileDict[detKeys[0]]))
                        sampling_strain=np.loadtxt(injectionFolder+'/'
                                                   +det+'/'+injectionFileDict[det][index_sample])
                        while (len(sampling_strain)/fs > maxDuration_
                               or index_sample in list(index_selection.values())):
                            index_sample=np.random.randint(0,len(injectionFileDict[detKeys[0]]))
                            sampling_strain=np.loadtxt(injectionFolder+'/'
                                                       +det+'/'+injectionFileDict[det][index_sample])
                        for det in detectors: index_selection[det]=index_sample
                    else:
                        index_sample=np.random.randint(0,len(injectionFileDict[detKeys[0]]))
                        for det in detectors: index_selection[det]=index_sample
                            
#                 inj_ind = np.random.randint(0,len(injectionFileDict[detKeys[0]]))

            SNR0_dict={}
            back_dict={}
            inj_fft_0_dict={}
            asd_dict={}
            PSD_dict={}
            gps_list = []
            
            
            
            for det in detKeys:
                
                # In case we want to put different injection to every detectors.
#                 if differentSignals==True:
#                     inj_ind = np.random.randint(0,len(injectionFileDict[detKeys[0]]))

                if backgroundType == 'optimal':

                    # Creation of the artificial noise.
                    PSD,X,T=simulateddetectornoise(profile[det],windowSize,fs,10,fs/2)
                    PSD_dict[det]=PSD
                    back_dict[det] = X
                    # Making the noise a TimeSeries
                    back=TimeSeries(X,sample_rate=fs) 
                    # Calculating the ASD so tha we can use it for whitening later
                    asd=back.asd(1,0.5)
                    asd_dict[det] = asd
                    gps_list.append(0.0)


                elif backgroundType == 'sudo_real':

                    noise_seg=noise_segDict[det]
                    # Calling the real noise segments
                    noise=noise_seg[ind[det][I]:ind[det][I]+windowSize*fs]  
                    # Generating the PSD of it
                    p, f = psd(noise, Fs=fs, NFFT=fs) 
                    p, f=p[1::],f[1::]
                    # Feeding the PSD to generate the sudo-real noise.            
                    PSD,X,T=simulateddetectornoise([f,p],windowSize,fs,10,fs/2)
                    PSD_dict[det]=PSD
                    # Making the noise a TimeSeries
                    back=TimeSeries(X,sample_rate=fs)
                    # Calculating the ASD so tha we can use it for whitening later
                    asd=back.asd(1,0.5)                 
                    asd_dict[det] = asd
                    gps_list.append(gps0+ind[det][I]/fs)
                    back_dict[det] = back.value
                    
                elif backgroundType == 'real':
                    noise_seg=noise_segDict[det]
                    # Calling the real noise segments
                    noise=noise_seg[ind[det][I]:ind[det][I]+windowSize*fs] 
                    # Calculatint the psd of FFT=1s
                    p, f = psd(noise, Fs=fs,NFFT=fs)
                    # Interpolate so that has t*fs values
                    psd_int=interp1d(f,p)                                     
                    PSD=psd_int(np.arange(0,fs/2,1/windowSize))
                    PSD_dict[det]=PSD
                    # Making the noise a TimeSeries
                    back=TimeSeries(noise,sample_rate=fs)
                    back_dict[det] = back
                    # Calculating the ASD so tha we can use it for whitening later
                    asd=back.asd(1,0.5)
                    asd_dict[det] = asd
                    gps_list.append(gps0+ind[det][I]/fs)

                #If this dataset includes injections:            
                if injectionFolder != None:
                    # Calling the templates generated with PyCBC
                    # OLD inj=load_inj(injectionFolder,injectionFileDict[det][inj_ind], det) 
                    inj = np.loadtxt(injectionFolder+'/'+det+'/'+injectionFileDict[det][index_selection[det]])

                    # Saving the length of the injection
                    inj_len = len(inj)/fs
                    inj_len_original=min(inj_len,duration)
                    # I put a random offset for all injection so that
                    # the signal is not always in the same place
                    if inj_len > duration: 
                        print('Injection was cropped symmetricaly by '
                              +str(inj_len_original-fs*duration)+' to fit duration.')

                        inj = inj[int((inj_len-duration)*fs/2):]
                        inj_len_=len(inj)/fs                    
                        inj = inj[:int(duration*fs)]
                        inj_len = len(inj)/fs
                        
                    if inj_len <= duration:
                        inj = np.hstack((np.zeros(int(fs*(duration-inj_len)/2)), inj))
                        inj_len_=len(inj)/fs                    
                        inj = np.hstack((inj, np.zeros(int(fs*(duration-inj_len_)))))
                        
                    # DISPOSITIONS
                    
                    # Default case when coherent
                    if disposition == 0:
                        if det == detKeys[0]:
                            disp = np.random.random_integers(low = min(-int(fs*(duration-inj_len_original)/2),0) 
                                                           ,high = max(int(fs*((duration-inj_len_original)/2 
                                                                         + injectionCrop*(duration))),1))
                            
#                     # Case when signals will be randomly positioned for each detector seperatly. 
#                     # Does not affect if signals will be different or not.
#                     elif isinstance(disposition,(list,tuple):
#                         disp = np.random.random_integers(low = min(-int(fs*(duration-inj_len)/2),0) 
#                                                          ,high = max(int(fs*((duration-inj_len)/2 
#                                                                          + injectionCrop*(duration-0.1))),1))
                    # Case when signals will be positioned within a duration stated by the disposition.
                    # The center of this duration will be moved randomly given the maxDuration
                    # of the injections.
                    elif isinstance(disposition_,(int,float)) and disposition_>0:
                        if det == detKeys[0]:
                            wind = 0.2*disposition_/max((len(detKeys)-1),1)
                            center_position = np.random.random_integers(
                                low = min(-int(fs*(duration-maxDuration_-disposition_)/2),0)
                               ,high = max(int(fs*(duration-maxDuration_-disposition_)/2)+1,1))
                                # the +1 is because there is an error if the high becomes 0
                                
                            if len(detKeys)>1:
                                positions=np.arange(-disposition_/2,disposition_/2+0.0001 
                                                    ,disposition_/max((len(detKeys)-1),1))
                                positions=list([positions[0]+np.random.rand()*wind/2]
                                     +list(p+np.random.rand()*wind-wind/2 for p in positions[1:-1])
                                     +[positions[-1]-np.random.rand()*wind/2])
                                
                            else:
                                positions=[0.0+np.random.rand()*wind/2]

                            np.random.shuffle(positions)
                        disp = center_position+int(positions[detKeys.index(det)]*fs)
                    
                    
                    if disp >= 0: 
                        inj = np.hstack((np.zeros(int(fs*(windowSize-duration)/2)),inj[disp:]
                                             ,np.zeros(int(fs*(windowSize-duration)/2)+disp)))   
                    if disp < 0: 
                        inj = np.hstack((np.zeros(int(fs*(windowSize-duration)/2)-disp),inj[:disp]
                                             ,np.zeros(int(fs*(windowSize-duration)/2)))) 
                    

                    # Calculating the one sided fft of the template,
                    inj_fft_0=np.fft.fft(inj)
                    inj_fft_0_dict[det] = inj_fft_0
                    
                    # we get rid of the DC value and everything above fs/2.
                    inj_fft_0N=np.abs(inj_fft_0[1:int(windowSize*fs/2)+1]) 
                    
                    SNR0_dict[det]=np.sqrt(param*2*(1/windowSize)*np.sum(np.abs(inj_fft_0N
                                  *inj_fft_0N.conjugate())[windowSize*fl-1:windowSize*fm-1]
                                /PSD_dict[det][windowSize*fl-1:windowSize*fm-1]))
                                        
                    if single == True:
                        if det != luckyDet:
                            SNR0_dict[det] = 0.01 # Making single detector injections as glitch
                            inj_fft_0_dict[det] = np.zeros(windowSize*fs)


                else:
                    SNR0_dict[det] = 0.01 # avoiding future division with zero
                    inj_fft_0_dict[det] = np.zeros(windowSize*fs)


            # Calculation of combined SNR    
            SNR0=np.sqrt(np.sum(np.asarray(list(SNR0_dict.values()))**2))

            # Tuning injection amplitude to the SNR wanted
            podstrain = []
            podPSD = []
            podCorrelations=[]
            SNR_new=[]
            psdPlugDict={}
            plugInToApply=[]

            for det in detectors:
                fft_cal=(injectionSNR/SNR0)*inj_fft_0_dict[det]         
                inj_cal=np.real(np.fft.ifft(fft_cal*fs))
                
                # Joining calibrated injection and background noise
                strain=TimeSeries(back_dict[det]+inj_cal,sample_rate=fs,t0=0).astype('float64') 
                # Bandpassing 
                strain=strain.bandpass(20,int(fs/2)-1)
                # Whitenning the data with the asd of the noise
                strain=strain.whiten(1,0.5,asd=asd_dict[det])
                # Crop data to the duration length
                strain=strain[int(((windowSize-duration)/2)*fs):int(((windowSize+duration)/2)*fs)]
                podstrain.append(strain.value.tolist())
                
                if 'snr' in plugins:
                    # Adding snr value as plugin.
                    # Calculating the new SNR which will be slightly different that the desired one.    
                    inj_fft_N=np.abs(fft_cal[1:int(windowSize*fs/2)+1]) 
                    SNR_new.append(np.sqrt(param*2*(1/windowSize)*np.sum(np.abs(inj_fft_N
                                *inj_fft_N.conjugate())[windowSize*fl-1:windowSize*fm-1]
                                /PSD_dict[det][windowSize*fl-1:windowSize*fm-1])))
                    
                    plugInToApply.append(PlugIn('snr'+det,SNR_new[-1]))

                if 'psd' in plugins:
                    podPSD.append(asd_dict[det]**2)
            
            if 'psd' in plugins:
                plugInToApply.append(PlugIn('psd'
                                            ,genFunction=podPSD
                                            ,plotFunction=plotpsd
                                            ,plotAttributes=['detectors','fs']))
                
            if 'correlation' in plugins:
                plugInToApply.append(knownPlugIns('correlation'))

            pod = DataPod(strain = podstrain
                               ,fs = fs
                               ,gps = gps_list
                               ,labels =  labels
                               ,detectors = detKeys
                               ,duration = duration)
            
            for pl in plugInToApply:
                pod.addPlugIn(pl)
                
            DATA.add(pod)
            
            
        random.shuffle(DATA.dataPods)
        if savePath!=None:
            DATA.save(savePath+'/'+name,'pkl')
        else:
            return(DATA)

        
        
        
        
           
        
        


import string

def auto_gen(duration 
             ,fs
             ,detectors
             ,size
             ,injectionFolder = None
             ,labels = None
             ,backgroundType = None
             ,injectionSNR = None
             ,firstDay = None
             ,windowSize = None #(32)            
             ,timeSlides = None #(1)
             ,startingPoint = None
             ,name = None
             ,savePath = None                  
             ,single = False  # Making single detector injections as glitch
             ,injectionCrop = 0   # Allowing part precentage of the injection to be croped
                                  # so that the signal can move from the center. Used only 
                                  # when injection duration = duaration
             ,disposition=None
             ,maxDuration=None
             ,differentSignals=False
             ,extras=None): 




    # ---------------------------------------------------------------------------------------- #
    # --- duration --------------------------------------------------------------------------- #

    if not (isinstance(duration,(float,int)) and duration>0 ):
        raise ValueError('The duration value has to be a possitive float'
            +' or integer representing seconds.')

    # ---------------------------------------------------------------------------------------- #    
    # --- fs - sample frequency -------------------------------------------------------------- #

    if not (isinstance(fs,int) and fs>0):
        raise ValueError('Sample frequency has to be a positive integer.')

    # ---------------------------------------------------------------------------------------- #    
    # --- detectors -------------------------------------------------------------------------- #

    if isinstance(detectors,(str,list)):
        for d in detectors:
            if d not in ['H','L','V','K','I']:
                raise ValueError("detectors have to be a list of strings or a string"+
                                " with at least one the followings as elements: \n"+
                                "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
                                "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
                                "\n'U' if you don't want to specify detector")
    else:
        raise ValueError("detectors have to be a list of strings or a string"+
                        " with at least one the followings as elements: \n"+
                        "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
                        "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
                        "\n'U' if you don't want to specify detector")
    if isinstance(detectors,str):
        detectors = list(detectors)

    # ---------------------------------------------------------------------------------------- #    
    # --- size ------------------------------------------------------------------------------- #        

    if not (isinstance(size, int) and size > 0):
        raise ValueError("size must be a possitive integer.")


    # ---------------------------------------------------------------------------------------- #
    # --- injectionFolder -------------------------------------------------------------------- #

    if injectionFolder == None:
        pass
    elif isinstance(injectionFolder, str):            
        if (('/' in injectionFolder) and os.path.isdir(injectionFolder)):
            injectionFolder_set = injectionFolder.split('/')[-1]
        elif (('/' not in injectionFolder) and any('injections' in p for p in sys.path)):
            for p in sys.path:
                if ('injections' in p):
                    injectionFolder_path = (p.split('injections')[0]+'injections'
                        +'/'+injectionFolder)
                    if os.path.isdir(injectionFolder_path):
                        injectionFolder = "'"+injectionFolder_path+"'"
                        injectionFolder_set = injectionFolder.split('/')[-1]
                    else:
                        raise FileNotFoundError('No such file or directory:'
                                                +injectionFolder_path)

        else:
            raise FileNotFoundError('No such file or directory:'+injectionFolder) 
    else:
        raise TypeError("cbcFolder has to be a string indicating a folder "
                        +"in MLyWorkbench or a full path to a folder")

    # ---------------------------------------------------------------------------------------- #    
    # --- labels ----------------------------------------------------------------------------- #

    if labels == None:
        labels = {'type' : 'UNDEFINED'}
        
    elif not isinstance(labels,dict):
        raise TypeError(" Labels must be a dictionary.Suggested keys for labels"
                        +"are the following: \n{ 'type' : ('noise' , 'cbc' , 'signal'"
                        +" , 'burst' , 'glitch', ...),\n'snr'  : ( any int or float number "
                        +"bigger than zero),\n'delta': ( Declination for sky localisation,"
                        +" float [-pi/2,pi/2])\n'ra'   : ( Right ascention for sky "
                        +"localisation float [0,2pi) )})")

    # ---------------------------------------------------------------------------------------- #    
    # --- backgroundType --------------------------------------------------------------------- #

    if backgroundType == None:
        backgroundType = 'optimal'
    elif not (isinstance(backgroundType,str) 
          and (backgroundType in ['optimal','sudo_real','real'])):
        raise ValueError("backgroundType is a string that can take values : "
                        +"'optimal' | 'sudo_real' | 'real'.")

    # ---------------------------------------------------------------------------------------- #    
    # --- injectionSNR ----------------------------------------------------------------------- #
    if injectionSNR == None :
        injectionSNR = [0]
        print("Injection SNR is None. Only one noise set will be created")
    if (isinstance(injectionSNR, (int,float)) and injectionSNR >= 0):
        injectionSNR = [injectionSNR]
    if not isinstance(injectionSNR, list):
        raise TypeError("InjectionSNR must be a list with SNR values of the sets. If you "
                        +"don't want injections just set the value to 0 for the sets")
    if (any(snr != 0 for snr in injectionSNR)  and  injectionFolder == None):
        raise ValueError("If you want to use an injection for generation of"+
                         "data, you have to specify the SNR you want and no zero.")
    if not all((isinstance(snr,(int,float)) and snr >=0) for snr in injectionSNR):
        raise ValueError("injectionSNR values have to be a positive numbers or zero")
    snr_list=[]
    c=0
    for snr in injectionSNR:
        if snr!=0:
            snr_list.append(snr)
        else:
            c+=1
            snr_list.append('No'+str(c))

    # ---------------------------------------------------------------------------------------- #    
    # --- firstDay --------------------------------------------------------------------------- #
    
    if backgroundType in ['sudo_real','real'] and firstDay == None:
        raise ValueError("You have to set the first day of the real data")
    elif backgroundType in ['sudo_real','real'] and isinstance(firstDay,str):
        if '/' in firstDay and os.path.isdir(firstDay):
            sys.path.append(firstDay.split(str(fs))[0]+'/'+str(fs)+'/')
            date_list_path=firstDay.split(str(fs))[0]+'/'+str(fs)
            firstDay = firstDay.split('/')[-1]
        elif '/' not in firstDay and os.path.isdir(firstDay):
            pass
        else:
            raise FileNotFoundError("No such file or directory:"+firstDay)
    elif backgroundType == 'optimal':
        pass
    else:
        raise TypeError("Path must be a string")
            
    
    # ---------------------------------------------------------------------------------------- #    
    # --- windowSize --(for PSD)-------------------------------------------------------------- #        

    if windowSize == None: windowSize = duration*8
    if not isinstance(windowSize,int):
        raise ValueError('windowSize needs to be an integral')
    if windowSize < duration :
        raise ValueError('windowSize needs to be bigger than the duration')

    # ---------------------------------------------------------------------------------------- #    
    # --- timeSlides ------------------------------------------------------------------------- #

    if timeSlides == None: timeSlides = 1
    if not (isinstance(timeSlides, int) and timeSlides >=1) :
        raise ValueError('timeSlides has to be an integer equal or bigger than 1')

    # ---------------------------------------------------------------------------------------- #    
    # --- startingPoint ---------------------------------------------------------------------- #

    if startingPoint == None : startingPoint = windowSize 
    if not (isinstance(startingPoint, int) and startingPoint >=0) :
        raise ValueError('timeSlides has to be an integer')        

    # ---------------------------------------------------------------------------------------- #    
    # --- name ------------------------------------------------------------------------------- #

    if name == None : name = ''
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')

    # ---------------------------------------------------------------------------------------- #    
    # --- savePath --------------------------------------------------------------------------- #

    if savePath == None : 
        savePath ='.'
    elif (savePath,str): 
        if not os.path.isdir(savePath) : 
            raise FileNotFoundError('No such file or directory:' +savePath)
    else:
        raise TypeError("Destination Path has to be a string valid path")
    if savePath[-1] == '/' : savePath=savePath[:-1]
    
    # ---------------------------------------------------------------------------------------- #    
               
    # The number of sets to be generated.
    num_of_sets = len(injectionSNR)

    # If noise is optimal it is much more simple
    if backgroundType == 'optimal':

        d={'size' : num_of_sets*[size]
           , 'start_point' : num_of_sets*[startingPoint]
           , 'set' : snr_list
           , 'name' : list(name+'_'+str(snr_list[i]) for i in range(num_of_sets))}

        print('These are the details of the datasets to be generated: \n')
        for i in range(len(d['size'])):
            print(d['size'][i], d['start_point'][i] ,d['name'][i])
                           
    # If noise is using real noise segments it is complicated   
    else:
        
        # Generating a list with all the available dates in the ligo_data folder
        date_list=dirlist( date_list_path)

        # The first day is the date we want to start using data.
        # Calculating the index of the initial date
        date=date_list[date_list.index(firstDay)]
        # All dates have an index in the list of dates following chronological order
        counter=date_list.index(firstDay)  
        
        # Calculation of the duration 
        # Here we infere the duration needed given the timeSlides used in the method
        
        # In this we just use the data as they are.
        if timeSlides==1:
            duration_need = size*num_of_sets*duration
            tail_crop=0
        # Timeslides of even numbers have a different algorithm that the odd number ones.
        if timeSlides%2 == 0:
            duration_need = ceil(size*num_of_sets/(timeSlides*(timeSlides-2)))*timeSlides*duration
            tail_crop=timeSlides*duration
        if timeSlides%2 != 0 and timeSlides !=1 :
            duration_need = ceil(size*num_of_sets/(timeSlides*(timeSlides-1)))*timeSlides*duration
            tail_crop=timeSlides*duration


        # Creation of lists that indicate characteristics of the segments based on the duration needed. 
        # These are used for the next step.

        # The following while loop checks and stacks durations of the data in date files. In this way
        # we note which of them are we gonna need for the generation.
        duration_total = 0
        duration_, gps_time, seg_list=[],[],[]

        while duration_need > duration_total:
            counter+=1
            segments=dirlist( date_list_path+'/'+date+'/'+detectors[0])
            print(date)
            for seg in segments:
                for j in range(len(seg)):
                    if seg[j]=='_': 
                        gps, dur = seg[j+1:-5].split('_')
                        break

                duration_.append(int(dur))
                gps_time.append(int(gps))
                seg_list.append([date,seg])

                duration_total+=(int(dur)-3*windowSize-tail_crop)
                print('    '+seg)

                if duration_total > duration_need: break

            if counter==len(date_list): counter==0        
            date=date_list[counter]


        # To initialise the generators, we will first create the code and the 
        # names they will have. This will create the commands that generate
        # all the segments needed.

        size_list=[]            # Sizes for each generation of noise 
        starting_point_list=[]  # Starting points for each generation of noise(s)
        seg_list_2=[]           # Segment names for each generation of noise
        number_of_set=[]        # No of set that this generation of noise will go
        name_list=[]            # List with the name of the set to be generated
        number_of_set_counter=0 # Counter that keeps record of how many 
        # instantiations have left to be generated 
                                # to complete a set


        set_num=0

        for i in range(len(np.array(seg_list)[:,1])):


            # local size indicates the size of the file left for generation 
            # of datasets, when it is depleted the algorithm moves to the next               
            # segment. Here we infere the local size given the timeSlides used in 
            # the method.

            if timeSlides==1:    # zero lag case
                local_size=int((duration_[i]-3*windowSize-tail_crop)/duration) #changed ceil to int
            if timeSlides%2 == 0:
                local_size=int((duration_[i]-3*windowSize-tail_crop)
                                /duration/timeSlides)*timeSlides*(timeSlides-2)
            if timeSlides%2 != 0 and timeSlides !=1 :
                local_size=int((duration_[i]-3*windowSize-tail_crop)
                                /duration/timeSlides)*timeSlides*(timeSlides-1)

            # starting point always begins with the window of the psd to avoid
            # deformed data of the begining    
            local_starting_point=windowSize

            # There are three cases when a segment is used.
            # 1. That it has to fill a previous set first and then move 
            # to the next
            # 2. That it is the first set so there is no previous set to fill
            # 3. It is too small to fill so its only part of a set.
            # Some of them go through all the stages

            if (len(size_list) > 0 and number_of_set_counter > 0 
                and local_size >= size-number_of_set_counter):

                # Saving the size of the generation
                size_list.append(size-number_of_set_counter) 
                # Saving the name of the date file and seg used
                seg_list_2.append(seg_list[i])          
                # Saving the startint_point of the generation
                starting_point_list.append(local_starting_point)
                # Update the the values for the next set
                local_size-=(size-number_of_set_counter)                     
                if timeSlides==1:
                    local_starting_point+=((size-number_of_set_counter)
                                           *duration)
                if timeSlides%2 == 0:
                    local_starting_point+=(ceil((size
                    -number_of_set_counter)/timeSlides/(timeSlides-2))*timeSlides*duration)
                if timeSlides%2 != 0 and timeSlides !=1 :
                    local_starting_point+=(ceil((size
                    -number_of_set_counter)/timeSlides/(timeSlides-1))*timeSlides*duration)

                number_of_set_counter += (size-number_of_set_counter)

                # If this generation completes the size of a whole set
                # (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(snr_list[set_num])
                    if size_list[-1]==size: 
                        name_list.append(name+'_'+str(snr_list[set_num]))
                    else:
                        name_list.append('part_of_'
                            +name+'_'+str(snr_list[set_num]))
                    set_num+=1
                    number_of_set_counter=0
                    if set_num >= num_of_sets: break

                elif number_of_set_counter < size:
                    number_of_set.append(snr_list[set_num])
                    name_list.append('part_of_'+name+'_'+str(snr_list[set_num]))

            if (len(size_list) == 0 or number_of_set_counter==0):

                while local_size >= size:


                    # Generate data with size 10000 with final name of 
                    # 'name_counter'
                    size_list.append(size)
                    seg_list_2.append(seg_list[i])
                    starting_point_list.append(local_starting_point)

                    #Update the the values for the next set
                    local_size -= size
                    if timeSlides==1: local_starting_point+=size*duration
                    if timeSlides%2 == 0: 
                        local_starting_point+=(ceil(size/timeSlides
                                                    /(timeSlides-2))*timeSlides*duration)
                    if timeSlides%2 != 0 and timeSlides !=1 :
                        local_starting_point+=(ceil(size/timeSlides
                                                    /(timeSlides-1))*timeSlides*duration)
                    number_of_set_counter+=size

                    # If this generation completes the size of a whole set
                    # (with size=size) it changes the labels
                    if number_of_set_counter == size:
                        #print(set_num)
                        number_of_set.append(snr_list[set_num])
                        if size_list[-1]==size: 
                            name_list.append(name+'_'+str(snr_list[set_num]))
                        else:
                            name_list.append('part_of_'
                                             +name+'_'+str(snr_list[set_num]))
                        set_num+=1
                        if set_num >= num_of_sets: break # CHANGED FROM > TO >= DUE TO ERROR
                        number_of_set_counter=0

            if (local_size < size and local_size >0 and set_num < num_of_sets):
                print(local_size, num_of_sets, set_num)

                # Generate data with size 'local_size' with local name to be
                # fused with later one
                size_list.append(local_size)
                seg_list_2.append(seg_list[i])
                starting_point_list.append(local_starting_point)

                # Update the the values for the next set
                number_of_set_counter+=local_size  

                # Saving a value for what is left for the next seg to generate
                # If this generation completes the size of a whole set
                # (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(snr_list[set_num])
                    if size_list[-1]==size: 
                        name_list.append(name+'_'+str(snr_list[set_num]))
                    else:
                        name_list.append('part_of_'
                                         +name+'_'+str(snr_list[set_num]))
                    set_num+=1
                    if set_num >= num_of_sets: break
                    number_of_set_counter=0

                elif number_of_set_counter < size:
                    number_of_set.append(snr_list[set_num])
                    name_list.append('part_of_'+name+'_'+str(snr_list[set_num]))


        d={'segment' : seg_list_2, 'size' : size_list 
           , 'start_point' : starting_point_list, 'set' : number_of_set
           , 'name' : name_list}

        print('These are the details of the datasets to be generated: \n')
        for i in range(len(d['segment'])):
            print(d['segment'][i], d['size'][i], d['start_point'][i] ,d['name'][i])
        
        
    answers = ['no','n', 'No','NO','N','yes','y','YES','Yes','Y','exit']
    answer = None
    while answer not in answers:
        print('Should we proceed to the generation of the following'
              +' data y/n ? \n \n')
        answer=input()
        if answer not in answers: print("Not valid answer ...")
    
    if answer in ['no','n', 'No','NO','N','exit']:
        print('Exiting procedure ...')
        return
    elif answer in ['yes','y','YES','Yes','Y']:
        print('Type the name of the dataset directory:')
        dir_name = '0 0'
        while not dir_name.isidentifier():
            dir_name=input()
            if not dir_name.isidentifier(): print("Not valid Folder name ...")
        
    path = savePath+'/'
    print("The current path of the directory is: \n"+path+dir_name+"\n" )  
    answer = None
    while answer not in answers:
        print('Do you accept the path y/n ?')
        answer=input()
        if answer not in answers: print("Not valid answer ...")

    if answer in ['no','n', 'No','NO','N','exit']:
        print('Exiting procedure ...')
        return
            
    elif answer in ['yes','y','YES','Yes','Y']:
        if os.path.isdir(path+dir_name):
            answer = None
            while answer not in answers:
                print('Already existing '+dir_name+' directory, do you want to'
                      +' overwrite it? y/n')
                answer=input()
                if answer not in answers: print("Not valid answer ...")
            if answer in ['yes','y','YES','Yes','Y']:
                os.system('rm -r '+path+dir_name)
            elif answer in ['no','n', 'No','NO','N']:
                print('Generation is cancelled\n')
                print('Exiting procedure ...')
                return
            
    print('Initiating procedure ...')
    os.system('mkdir '+path+dir_name)
    
    error = path+dir_name+'/condor/error'
    output = path+dir_name+'/condor/output'
    log = path+dir_name+'/condor/log'
    submit = path+dir_name+'/condor/submit'

    dagman = Dagman(name='falsAlarmDagman',
            submit=submit)
    job_list=[]
    
    print('Creation of directory complete: '+path+dir_name)
    #os.system('cd '+path+dir_name)
                

    for i in range(len(d['size'])):

        with open(path+dir_name+'/'+'gen_'+d['name'][i]+'_'
            +str(d['size'][i])+'.py','w') as f:
            f.write('#! /usr/bin/env python3\n')
            f.write('import sys \n')
            #This path is used only for me to test it
            pwd=os.getcwd()
            if 'vasileios.skliris' in pwd:
                f.write('sys.path.append(\'/home/vasileios.skliris/mly/\')\n')

            f.write('from mly.datatools import DataPod, DataSet\n\n')

            if isinstance(d['set'][i],(float,int)):
                token_snr = str(d['set'][i])
            else:
                token_snr = '0'
            f.write("import time\n\n")
            f.write("t0=time.time()\n")
            
            if injectionFolder!=None and injectionFolder[0]!="'":
                injectionFolder = "'"+injectionFolder+"'"
            if backgroundType == 'optimal':
                comand=( "SET = DataSet.generator(\n"
                         +24*" "+"duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         +24*" "+",size = "+str(d['size'][i])+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",injectionFolder ="+str(injectionFolder)+"\n"
                         +24*" "+",labels = "+str(labels)+"\n"
                         +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                         +24*" "+",injectionSNR = "+token_snr+"\n"
                         +24*" "+",name = '"+str(d['name'][i])+"_"+str(d['size'][i])+"'\n"
                         +24*" "+",savePath ='"+path+dir_name+"'\n"
                         +24*" "+",single = "+str(single)+"\n"
                         +24*" "+",injectionCrop = "+str(injectionCrop)+"\n"
                         +24*" "+",differentSignals = "+str(differentSignals)+"\n"
                         +24*" "+",extras = "+str(extras)+")\n")

            else:
                f.write("sys.path.append('"+date_list_path[:-1]+"')\n")
                comand=( "SET = DataSet.generator(\n"
                         +24*" "+"duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         +24*" "+",size = "+str(d['size'][i])+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",injectionFolder = "+str(injectionFolder)+"\n"
                         +24*" "+",labels = "+str(labels)+"\n"
                         +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                         +24*" "+",injectionSNR = "+token_snr+"\n"
                         +24*" "+",noiseSourceFile = "+str(d['segment'][i])+"\n"
                         +24*" "+",windowSize ="+str(windowSize)+"\n"
                         +24*" "+",timeSlides ="+str(timeSlides)+"\n"
                         +24*" "+",startingPoint = "+str(d['start_point'][i])+"\n"
                         +24*" "+",name = '"+str(d['name'][i])+"_"+str(d['size'][i])+"'\n"
                         +24*" "+",savePath ='"+path+dir_name+"'\n"
                         +24*" "+",single = "+str(single)+"\n"
                         +24*" "+",injectionCrop = "+str(injectionCrop)+"\n"
                         +24*" "+",differentSignals = "+str(differentSignals)+"\n"
                         +24*" "+",extras = "+str(extras)+")\n")

            
            f.write(comand+'\n\n')
            f.write("print(time.time()-t0)\n")

        os.system('chmod 777 '+path+dir_name+'/'+'gen_'+d['name'][i]+'_'
            +str(d['size'][i])+'.py' )
        job = Job(name='partOfGeneration_'+str(i)
                  ,executable=path+dir_name+'/'+'gen_'+d['name'][i]+'_'+str(d['size'][i])+'.py' 
               ,submit=submit
               ,error=error
               ,output=output
               ,log=log
               ,getenv=True
               ,dag=dagman
               ,extra_lines=["accounting_group_user=vasileios.skliris"
                             ,"accounting_group=ligo.dev.o3.burst.grb.xoffline"] )

        job_list.append(job)

    with open(path+dir_name+'/info.txt','w') as f3:
        f3.write('INFO ABOUT DATASETS GENERATION \n\n')
        f3.write('fs: '+str(fs)+'\n')
        f3.write('duration: '+str(duration)+'\n')
        f3.write('window: '+str(windowSize)+'\n')
        if injectionFolder!=None:
            f3.write('injectionFolder: '+str(injectionFolder)+'\n')
        
        if backgroundType != 'optimal':
            f3.write('timeSlides: '+str(timeSlides)+'\n'+'\n')
            for i in range(len(d['size'])):
                f3.write(d['segment'][i][0]+' '+d['segment'][i][1]
                         +' '+str(d['size'][i])+' '
                         +str(d['start_point'][i])+'_'+d['name'][i]+'\n')
            
    with open(path+dir_name+'/final_gen.py','w') as f4:
        f4.write("#! /usr/bin/env python3\n")
        pwd=os.getcwd()
        if 'vasileios.skliris' in pwd:
            f4.write("import sys \n")
            f4.write("sys.path.append('/home/vasileios.skliris/mly/')\n")
        f4.write("from mly.datatools import *\n")
        f4.write("finalise_gen('"+path+dir_name+"')\n")
        
    os.system('chmod 777 '+path+dir_name+'/final_gen.py')
    final_job = Job(name='finishing'
               ,executable=path+dir_name+'/final_gen.py'
               ,submit=submit
               ,error=error
               ,output=output
               ,log=log
               ,getenv=True
               ,dag=dagman
               ,extra_lines=["accounting_group_user=vasileios.skliris"
                             ,"accounting_group=ligo.dev.o3.burst.grb.xoffline"] )
    
    final_job.add_parents(job_list)
    
    print('All set. Initiate dataset generation y/n?')
    answer4=input()

    if answer4 in ['yes','y','YES','Yes','Y']:
        print('Creating Job queue')
        
        dagman.build_submit()

        return
    
    else:
        print('Data generation canceled')
        os.system('cd')
        os.system('rm -r '+path+dir_name)
        return


    
def finalise_gen(path,generation=True):
    
    if path[-1]!='/': path=path+'/' # making sure path is right
    files=dirlist(path)             # making a list of files in that path 
    merging_flag=False              # The flag that makes the fusion to happen

    print('Running diagnostics for file: '+path+'  ... \n') 
    pyScripts=[]
    dataSets=[]
    for file in files:
        if (file[-3:]=='.py') and ('gen_' in file):
            pyScripts.append(file)
        if file[-4:]=='.pkl': 
            dataSets.append(file)

    # Checking if all files that should have been generated 
    # from auto_gen are here
    if len(dataSets)==len(pyScripts):
        print('Files succesfully generated, all files are here')
        print(len(dataSets),' out of ',len(pyScripts))
        merging_flag=True  # Declaring that merging can happen now
    
    # If some files haven't been generated it will show a failing message
    # with the processes that failed
    else:
        failed_pyScripts=[]
        for i in range(len(pyScripts)):
            pyScripts_id=pyScripts[i][4:-3]
            counter=0
            for dataset in dataSets:
                if pyScripts_id in dataset:
                    counter=1
            if counter==0:
                print(pyScripts[i],' failed to proceed')
                failed_pyScripts.append(pyScripts[i])
                
                
    if merging_flag==False and generation==True:
        
        with open(path+'/'+'flag_file.sh','w+') as f2:
             f2.write('#!/usr/bin/bash +x\n\n')
        print(path)
        error = path+'condor/error'
        output = path+'condor/output'
        log = path+'condor/log'
        submit = path+'condor/submit'

        repeat_dagman = Dagman(name='repeat_genDagman',
                submit=submit)
        repeat_job_list=[]
        
        if os.path.isfile(path+'/'+'auto_gen_redo.sh'):
            print("\nThe following scripts failed to run trough:\n")
            for script in failed_pyScripts:
                    
                repeat_job = Job(name='partOfGeneration_'+str(i)
                           ,executable=path+script
                           ,submit=submit
                           ,error=error
                           ,output=output
                           ,log=log
                           ,getenv=True
                           ,dag=repeat_dagman
                           ,extra_lines=["accounting_group_user=vasileios.skliris"
                                         ,"accounting_group=ligo.dev.o3.burst.grb.xoffline"] )

                repeat_job_list.append(repeat_job)

               
        repeat_final_job = Job(name='repeat_finishing'
                           ,executable=path+'finalise_gen.py'
                           ,submit=submit
                           ,error=error
                           ,output=output
                           ,log=log
                           ,getenv=True
                           ,dag=repeat_dagman
                           ,extra_lines=["accounting_group_user=vasileios.skliris"
                                         ,"accounting_group=ligo.dev.o3.burst.grb.xoffline"] )

        repeat_final_job.add_parents(repeat_job_list)
        
        repeat_dagman.build_submit()
            

    if merging_flag==True:
        
        setNames=[]
        setIDs=[]
        setSizes=[]
        finalNames=[]
        IDs,new_dat=[],[]
        for dataset in dataSets:
            if 'part_of' in dataset:
                setNames.append(dataset)
                setIDs.append(dataset.split('_')[-2])
                setSizes.append(dataset.split('_')[-1].split('.')[0])
                finalNames.append('_'.join(dataset.split('part_of_')[1].split('_')[:-2]))

                if dataset.split('_')[-2] not in IDs:
                    IDs.append(dataset.split('_')[-2])

        # Creating the inputs for the function data_fusion
        for k in range(len(IDs)):
            fusionNames=[]
            sizes = []
            for i in range(len(setNames)):
                if setIDs[i]==IDs[k]:
                    fusionNames.append(path+setNames[i])
                    sizes.append(int(setSizes[i]))
                    finalName = finalNames[i]+'_'+str(IDs[k])+'_'
            _=DataSet.fusion(fusionNames, sizes = sizes, save = path+finalName+str(sum(sizes)))

        # Deleting unnescesary file in the folder
        for file in dirlist(path):
            if (('.out' in file) or ('.py' in file)
                or ('part_of' in file) or ('.sh' in file)):
                os.system('rm '+path+file)

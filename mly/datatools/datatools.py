from .core import DataPodBase, DataSetBase
from ..tools import dirlist, internalLags, correlate
from ..plugins import *

import pickle
import os
import sys
import time
import random
import copy

import numpy as npl
import matplotlib.pyplot as plt

from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment,SegmentList,DataQualityFlag
from gwpy.time import to_gps, from_gps



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
            with open(name+'.pkl', 'wb') as output:
                pickle.dump(self, output, 4)

        elif saving_format == 'txt' and (saving_format in allowed_formats):
            np.savetxt(name+'.txt', self.strain)

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


            
            
            

              
class DataSet(DataSetBase):
    
    """DataSet is an object that helps manipulate groups of DataPods as a whole.
    The main attribute is a list of the DataPods. All methods are providing ways
    to manipulate the data and export them to desired shapes. 
    
    Attributes
    ----------
    
    dataPods: list of DataPod objects (optional)
        List of the DataPod objects to be part of this DataSet instance. All 
        DataPods are checked for inconsistences such as different shapes, 
        different number of detectors and different sample frequencies.
    
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
            finalName = name
        if type_ == 'pkl':
            if finalName[-4:]!='.pkl':
                finalName+='.pkl'
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
            if self.dataPods==[]:
                pod0 = newData
            else:
                pod0=self.dataPods[0]
            if newData.shape != pod0.shape:
                print("Pods with different shapes")
            elif newData.fs != pod0.fs:
                print("Pods woth different sample frequencies")
            else:
                self.dataPods.append(newData)
        elif isinstance(newData, DataSet):
            pod0=self.dataPods[0]
            for pod in newData:
                if pod.shape != pod0.shape:
                    print("Pods with different shapes")
                if pod.fs != pod0.fs:
                    print("Pods woth different sample frequencies")
                if not all(d in pod0.detectors for d in pod.detectors):
                    print("Pods with different detectors")
                else:
                    pass
            self.dataPods += newData.dataPods

        else:
            raise TypeError("Appended object is not a DataPod or Dataset")
            
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
            del set_

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
            raise ValueError('PlugIn '+str(plugin)+' not present in DataPod')
            
        if shape == None:
            shape = goods.shape
            shape = tuple([None]+list(shape[1:]))
        # print("goods.shape",goods.shape)
        if isinstance(shape,tuple):
            if all(((dim in goods.shape) or dim==None) for dim in shape):
                shapeList = list(shape)
                goodsShapeList = [None]+list(goods.shape)[1:]
                newIndex = list(shapeList.index(goodsShapeList[i]) for i in range(len(shape)))
                goods = np.transpose(goods, newIndex)
            else:
                raise ValueError("Shape values are not the same as the DataSet shape")

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
        print("Labels "+str(list(args))+" with shape "+str(goods.shape)+" are exported")
        return goods
   
    def exportGPS(self):
        goods =[]
        for pod in self.dataPods:
            goods.append(pod.gps)
        return goods
        
 
    def stackDetector(self,**kwargs):
        kwargs['size']=len(self)
        if 'duration' not in kwargs: kwargs['duration']=self[0].duration
        if 'fs' not in kwargs: kwargs['fs']=self[0].fs   
        if 'detectors' not in kwargs: 
            raise ValueError("You need to at least specify a detector")

        if 'plugins' not in kwargs: kwargs['plugins']=[]
        if 'psd' in self[0].pluginDict and 'psd' not in kwargs['plugins']: kwargs['plugins'].append('psd')
        # if 'snr'+self[0].detectors[0] in self[0].pluginDict and 'snr' not in kwargs['plugins']: 
        #     kwargs['plugins'].append('snr')
        

        newSet=generator(**kwargs)

        for i in range(len(self)):
            self[i].strain=np.vstack((self[i].strain,newSet[i].strain))
            self[i].detectors+=newSet[i].detectors
            self[i].gps+=newSet[i].gps
            if 'psd' in kwargs['plugins']: self[i].psd+=newSet[i].psd
            # if 'snr' in kwargs['plugins']:
            #     for d in newSet[i].detectors:
            #         self[i].addPlugIn(newSet[i].pluginDict['snr'+d])        
            if 'correlation' in self[i].pluginDict:
                self[i].addPlugIn(self[i].pluginDict['correlation'])   


# -*- coding: utf-8 -*-

from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment, SegmentList
import numpy as np
import pandas as pd
from sys import getsizeof
import copy
import pickle 
import os

################################################################################

class DataPodBase:
    # --- INFO --------------------------------------------------------------- #
    #                                                                          # 
    # DataPod is an object that holds all the information required from a data #
    # instance. The reason for this object is to be the building block for a   #
    # versatile dataset object that can be easyly used as input to many types  #
    # of machine learning networks.                                            # 
    #                                                                          # 
    # ------------------------------------------------------------------------ #
    
    #                                                                          #             
    ###--- STR & REPR methods -----------------------------------------------###        
    #                                                                          #
    
    def __str__(self):
        statement=''
        
        lim = 10
        labels_len=0
        final_len=81
        
        for _ in self._labels.keys():
            labels_len += len((str(self._labels[_])+' ')) 
            
        while final_len>80:
            labels_len=0
            for _ in self._labels.keys():
                labels_len += len((str(self._labels[_])+' '))
            final_len =(33+len(str(self.shape))+labels_len+4*lim)
            lim-=1    
            
        for i in range(len(self._detectors)):

            labels_str=''
            if i==len(self._detectors)-1 :
                labels_str+='  LABELS: '
                for _ in self._labels.keys():
                    labels_str += (str(self._labels[_])+' ') 

            info=''
            if i==0: 
                info+=str(self.shape)
            else:
                info+=len(str(self.shape))*' '
            info +=('[ '+str(self._strain[i][0])[0:lim]+' '
                    +str(self._strain[i][1])[0:lim]+' ... '
                    +str(self._strain[i][-2])[0:lim]+' '
                    +str(self._strain[i][-1])[0:lim]+' ] ')
            if self._detectors[i]!='U': info+=(self._detectors[i]) 
            if self._gps[i]!=0.0 : info+=('_'+str("%10d" % self._gps[i]))

            statement+=info+labels_str+' \n'

        return(statement)
    
    def __repr__(self):
        statement=''
        lim=6    
        for i in range(len(self._detectors)):

            labels_str=''
            if i==len(self._detectors)-1 :
                labels_str+='  LABELS: '
                for _ in self._labels.keys():
                    labels_str += (str(self._labels[_])+' ') 

            info=''
            if i==0: 
                info+=str(self.shape)
            else:
                info+=len(str(self.shape))*' '
            info +=('[ '+str(self._strain[i][0])[0:lim]+' '
                    +str(self._strain[i][1])[0:lim]+' ... '
                    +str(self._strain[i][-2])[0:lim]+' '
                    +str(self._strain[i][-1])[0:lim]+' ] ')
            if self._detectors[i]!='U': info+=(self._detectors[i]) 
            if self._gps[i]!=0.0 : info+=('_'+str("%010d" % self._gps[i]))

            statement+=info+labels_str+' \n'

        return(statement)
   
        
    #                                                                          #
    ###--- INIT method ------------------------------------------------------###
    #                                                                          #
    
    def __init__(self
                 , strain             # The core data (any consistent shape)
                 , fs = None          # Sample frequency
                 , gps = None         # GPS time of the data
                 , labels = None      # Labels of any type
                 , detectors = None   # Detectors coresponding to data
                 , duration = None    # Duration of the data
                 , metadata = None):  # Any other useful information 
        self.strain=strain
        self.fs=fs
        self.labels=labels
        self.detectors=detectors
        self.gps=gps
        self.duration=duration
        self.metadata=metadata
        self.pluginDict={}  

    #                                                                          #
    ### --- setters and getters ---------------------------------------------###   
    #                                                                          #


    @property
    def strain(self):
        return self._strain                 
    @strain.setter
    def strain(self,strain):
        
        # # Checking the type of input for strain:
        # ---> This is the type would want.
        if (isinstance(strain,np.ndarray)): 
            pass
        # ---> If it is a list we just make it a numpy.ndarray.
        elif (isinstance(strain,list)): 
            strain = np.array(strain)      
        # ---> If it is a TimeSeries we take the value only.   
        elif (isinstance(strain,TimeSeries)): 
            strain = strain.value      
        else:
            raise TypeError("strain type can be only one of the following:/n"+
                           "list \nnumpy.ndarray\ngwpy.timeseries.TimeSeries")

        # # Strain has to have clear fixed dimentions.
        # ---> If all elements have the same len with the first we are fine.
        if (isinstance(strain[0],np.ndarray) 
            and not all(len(x)==len(strain[0]) for x in strain)):
            raise IndexError("Some detector strain have different size that "+
                             "the others")

        # # Strain has to be free from infs and nans
        # ---> If there is an inf or nan in the strain an erros is raised.
        if not np.isfinite(strain).all():
            raise ValueError("There is nan or inf value in the strain")
        if all(np.isreal(x) for x in strain.flatten()):
            # # For one dimentional data, we need to specify the shape differently.   
            if len(strain.shape) == 1:
                strain= strain.reshape(1,-1)
            self._strain = strain
        else:
            raise TypeError("strain values can only be int or float")
            
    @property
    def fs(self):
        return self._fs
    
    @fs.setter
    def fs(self,fs):
        # # fs has to be a valid number.
        # ---> If fs is not a positive int and error is raised.
        if fs==None: fs = 1
        if isinstance(fs, int) and fs > 0:
            self._fs = fs 
        else:
            raise ValueError("Sample frequency must have no decimal part")
            

            
            

    @property
    def labels(self):
        return self._labels
    @labels.setter        
    def labels(self, labels):
        # # Labels are optional and it has to be a dictionary.
        # ---> If empty a label 'UNDEFINED' is given.
        if labels == None:
            labels={'type':'UNDEFINED'}
        # ---> If a dictionary is given we have to make sure is not empty.
        if isinstance(labels,dict):
            if len(labels) == 0:
                # ---> If it is empty we give the 'UNDEFINED' label.
                print("Labels should have at least one element key \'type\'."
                      +"\'UNDEFINED\' value has been set as \'type\'. ")
                self._labels={'type':'UNDEFINED'} 
            else:
                self._labels=labels
        # ---> If a non dictionary object is given, an error is called.
        else:
            raise TypeError("labels must be a dictionary")

    @property
    def detectors(self):
        return self._detectors
    @detectors.setter
    def detectors(self,detectors):
        
        
        # # Detectors is an optional parameter.
        # ---> If not given it will be assumed from the smallest dimention.
        # ---> This can create problems for dimentions > 1D.
        if detectors == None: 
            d=np.min(self._strain.shape)
            detectors = d*['U']
        if isinstance(detectors,str):
            detectors=list(d for d in detectors)
        # # If detectors is used, it has to be a list.
        # ---> If it is not an error is raised.
        if not isinstance(detectors,list): raise TypeError("Given type:"+str(type(detectors))+
            ". Detectors have to be a list of strings"+
            " with at least one the followings as elements: \n"+
            "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
            "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
            "\n'U' if you don't want to specify detector")
            
        # # Moreover it has to be a list containing specific detector initials.
        # ---> If elements of detectors aren't in detector it gives an error. 
        detectorOptions = ['H','L','V','K','I','U']
        if isinstance(detectors,list):
            for det in detectors:
                if det not in detectorOptions: 
                    raise ValueError("Given value:"+str(detectors)+
                        ". Detectors have to be a list of strings"+
                        " with at least one the followings as elements: \n"+
                        "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
                        "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India"+
                        " (INDIGO) \n"+
                        "\n'U' if you don't want to specify detector")
        
        # # The number of detectors must be shown at the first dimention.
        if (self.strain.shape[0] == len(detectors)):
            self._detectors = detectors
        # ---> If it is not in the first dimention maybe needs transpose.
        elif (self.strain.shape[0] != len(detectors)):
            # ---> If the number of detectors is present at some dimention.
            if ((len(detectors) in self.strain.shape) 
                and self.strain.shape.count(len(detectors))==1):
                # --- Then we traspose the strain.
                newShape = list(self.strain.shape)
                newShape.pop(newShape.index(len(detectors)))
                newShape.insert(0,len(detectors))
                ind = [list(self.strain.shape).index(i) for i in newShape]
                self.strain = np.transpose(self.strain, ind)
                print("Needed shape change of the strain occured")
                self._detectors = detectors
            # ---> If not, an error occures.
            else:
                raise IndexError("detectors must be as many as the strain")

    @property
    def gps(self):
        return self._gps  
    @gps.setter
    def gps(self,gps):
        # # GPS is optional or a list of positive numbers. 
        # ---> If it is a list, it has to include positive numbers.
        if gps==None:
            gps=[0]*len(self.detectors)
        if ((isinstance(gps,list) and len(gps)==len(self.detectors)) and
            all((isinstance(_,(float,int)) and _ >=0) for _ in gps)):
                self._gps = gps
        # ---> If not an error is raised.
        else:
            raise TypeError("gps has to be a list of positive numbers"
                            +" with number of elements equal to detectors")
    @property
    def duration(self):
        return self._duration
    @duration.setter
    def duration(self,duration):
        if duration == None:
            # ---> Safe assumption can happen only on timeseries (2D).
            if len(self.strain.shape) == 2:
                self._duration = self.strain.shape[1]/int(self.fs)
            # ---> If not definition is mandatory.
            else:
                raise TypeError("Duration of the input cannot be assumed ")
        # ---> If it is defined, it has to be a number.
        elif isinstance(duration,(int, float)):
            self._duration = duration
        # ---> If not an error is raised.
        else:
            raise TypeError("duration must be a float or int")

    @property
    def metadata(self):
        return self._metadata
    @metadata.setter
    def metadata(self,metadata):
        # # Metadata is optional and is a dictionary with exta information.
        if metadata == None:
            self._metadata = None
        # ---> If defined it has to be a non empty dictionary.
        elif isinstance(metadata,dict):
            if len(metadata.keys())!=0:
                self._metadata = metadata
            else: raise ValueError("metadata cannot be an empty dictionary")
        # ---> If not an error is raised.
        else:
            raise TypeError("metadata has to be a dictionary")
    
    # --- extra usefull attributes ------------------------------------------- #
    
    @property
    def shape(self):
        return self._strain.shape
                  
    @property
    def memory(self):
        return self._strain.nbytes
    
    # --- __MAGIC_METHODS__ -------------------------------------------------- #

    def __add__(self, num):
        newPod = copy.copy(self)
        if isinstance(num,(float,int)):
            if not np.isfinite(num).all():
                raise ValueError("inf or nan value is present")
            else:
                newPod._strain = newPod._strain + num
                return(newPod)
        elif isinstance(num,np.ndarray):
            if not np.isfinite(num).all():
                raise ValueError("inf or nan value is present")
            if (num.shape == self._strain.shape):
                newPod._strain = newPod._strain + num
                return(newPod)
            else:
                raise IndexError("You cannot operate shape "+str(num.shape)
                                +" with shape "+str(self._strain.shape))
        else:
            raise TypeError("Mathematical operations are not allowed with "
                           +str(type(num))+"types. Only int, float or" 
                           +" numpy.ndarray")
            
    def __iadd__(self, num):
        return(self.__add__(num))
    def __radd__(self, num):
        return(self.__add__(num))        
    def __sub__(self, num):
        return(self.__add__(-num))      
    def __isub__(self, num):
        return(self.__add__(-num))
    def __rsub__(self, num):
        return(self.__add__(-num))
        
    def __mul__(self, num):
        newPod = copy.copy(self)
        if isinstance(num,(float,int)):
            if not np.isfinite(num).all():
                raise ValueError("inf or nan value is present")
            else:
                newPod._strain = newPod._strain * num
                return(newPod)
        elif isinstance(num,np.ndarray):
            if not np.isfinite(num).all():
                raise ValueError("inf or nan value is present")
            if (num.shape == self._strain.shape):
                newPod._strain = newPod._strain * num
                return(newPod)
            else:
                raise IndexError("You cannot operate shape "+str(num.shape)
                                +" with shape "+str(self._strain.shape))
        else:
            raise TypeError("Mathematical operations are not allowed with "
                           +str(type(num))+"types. Only int, float or" 
                           +" numpy.ndarray")
    def __imul__(self, num):
        return(self.__mul__(num))
    def __rmul__(self, num):
        return(self.__mul__(num))
    

    def __truediv__(self, num):
        newPod = copy.copy(self)
        if isinstance(num,(float,int)):
            if not np.isfinite(num).all():
                raise ValueError("inf or nan value is present")
            elif num == 0:
                raise ValueError("Division with zero encountered")                
            else:
                newPod._strain = newPod._strain / num
                return(newPod)
        elif isinstance(num,np.ndarray):
            if not np.isfinite(num).all():
                raise ValueError("inf or nan value is present")
            elif (num==0).any():
                raise ValueError("Division with zero encountered") 
            if (num.shape == self._strain.shape):
                newPod._strain = newPod._strain / num
                return(newPod)
            else:
                raise IndexError("You cannot operate shape "+str(num.shape)
                                +" with shape "+str(self._strain.shape))
        else:
            raise TypeError("Mathematical operations are not allowed with "
                           +str(type(num))+"types. Only int, float or" 
                           +" numpy.ndarray")
            
    def __itruediv__(self, num):
        return(self.__truediv__(num))


    
    def __len__(self):
        return(len(self._strain))
        
    # def __getitem__() --> It is moved to the main function DataPod

    
    def __delitem__(self,key):
        if isinstance(key,int):
            del self._strain[key]
            del self._detectors[key]
            del self._gps[key]
        elif (isinstance(key,str) and (key in self._detectors) 
             and key!='U'):
            ind = self._detectors.index(key)
            del self._strain[ind]
            del self._detectors[ind]
            del self._gps[ind]
        else:
            raise TypeErrro("index can be int or a detector name that exists")
    def __copy__(self):
        return self
    #def __iter__(self)
    #def __contains__(self,item)


        
        
        
class DataSetBase:
    
    #                                                                          #             
    ###--- STR & REPR methods -----------------------------------------------###        
    #                                                                          #
    
    def __str__(self):
        statement=(str(len(self))+" instances with shape "
                   +str(self.dataPods[0].shape)+"\n\n")
        if len(self) > 5:
            statement = statement + str(self._dataPods[0])+ "\n"
            statement = statement + str(self._dataPods[1])+ "\n"
            statement = statement + (9*" "+6*("."+9*" ")+".\n")
            statement = statement + (9*" "+6*("."+9*" ")+".\n")
            statement = statement + (9*" "+6*("."+9*" ")+".\n\n")
            statement = statement + str(self._dataPods[-2])+ "\n"
            statement = statement + str(self._dataPods[-1])+ "\n"            
        else:
            for pod in self._dataPods:
                statement = statement + str(pod)      
        return (statement)
    
    def __repr__(self):
        statement=(str(len(self))+" instances with shape "
                   +str(self.dataPods[0].shape)+"\n\n")
        if len(self) > 5:
            statement = statement + str(self._dataPods[0])+ "\n"
            statement = statement + str(self._dataPods[1])+ "\n"
            statement = statement + (9*" "+6*("."+9*" ")+".\n")
            statement = statement + (9*" "+6*("."+9*" ")+".\n")
            statement = statement + (9*" "+6*("."+9*" ")+".\n\n")
            statement = statement + str(self._dataPods[-2])+ "\n"
            statement = statement + str(self._dataPods[-1])+ "\n"            
        else:
            for pod in self._dataPods:
                statement = statement + str(pod)      
        return (statement)
    
    #                                                                          #             
    ###--- INIT method ------------------------------------------------------###        
    #                                                                          #
    
    def __init__(self, dataPods=None, name = None):
        
        self.dataPods = dataPods
        self.name = name
        self.pluginDict={} 

    #                                                                          #
    ### --- setters and getters ---------------------------------------------###
    #                                                                          #

    @property
    def dataPods(self):
        return self._dataPods       
    
    @dataPods.setter
    def dataPods(self,dataPods):
        
        if isinstance(dataPods,list) and all(isinstance(x 
                                    , DataPodBase) for x in dataPods):
            pass
        elif (dataPods == None):
            dataPods = []
        else:
            raise TypeError("DataSet object has to be a list of DataPods")

        # pods similarity check --
        if len(dataPods) !=0:
            pod0=dataPods[0]
            for pod in dataPods:
                if pod.shape != pod0.shape:
                    print("Pods with different shapes")
                if pod.fs != pod0.fs:
                    print("Pods with different sample frequencies")
                if not all(d in pod0.detectors for d in pod.detectors):
                    print("Pods with different detectors")
        
        self._dataPods=dataPods
                
                    
                    
                    

    @property
    def name(self):
        return self._name 

    @name.setter
    def name(self,name):
        if name == None:
            name = 'NoName'
        if isinstance(name, str):
            self._name = name
        else:
            raise TypeError("name must be a str")    
            

    def duration(self):
        return self.dataPods[0].duration
        
    # def __getitem__() --> It is moved to the main function DataSets
    
    def __len__(self):
        return(len(self._dataPods))

    def selfcheck(self):
        pod0=self.dataPods[0]
        check = True
        for pod in self.dataPods:
            if pod.shape != pod0.shape:
                print("Pods with different shapes")
                check = False
            if pod.fs != pod0.fs:
                print("Pods with different sample frequencies")
                check = False
            if not all(d in pod0.detectors for d in pod.detectors):
                print("Pods with different detectors")
        if check == True:
            print("All good")
        return(check)
                      
        
# COMMON METHODS


import numpy as np
import matplotlib.pyplot as plt
from math import ceil 
from gwpy.timeseries import TimeSeries
from gwpy.io.kerberos import kinit
from gwpy.segments import DataQualityFlag
from gwpy.segments import Segment,SegmentList
from gwpy.time import to_gps
from gwpy.time import from_gps
from dqsegdb2.query import query_segments
import os
import time
import pandas as pd
################################################################################

def dirlist(filename,exclude=None):   
    """
    Function that returns the a list of the files
    in a directory with options for exclutions.

    Parameters
    ----------
    filename : `str`
        Name of the directory which contents we wan
        put on a list.

    exclude : `list` [`str`] (optional)
        A list of all the stings that if any of them 
        is present to the name of a file, it will not 
        include that file to the output.
    
    Returns
    -------
    file list : `list` [`str`]
        A list with all the filenames in the specified directory.
    
    
    """
    fn=os.listdir(filename) 
    if exclude==None:
        exclusions=[]
    else:
        exclusions=exclude
    fn_clean=[]
    for i in range(0,len(fn)): 
        if fn[i][0]!='.' and all(ex not in fn[i] for ex in exclusions):
            fn_clean.append(fn[i])
    fn_clean.sort()
    return fn_clean

def lcm(*args):
    """
    Function that returns the lowest common multiple 
    of all integers present in the arguments.

    Parameters
    -------------
    *arg : `int`
        Integer numbers of interest
    
    Returns
    -------
    lowest common multiple : `int`
        The lowest common multiple of the specified numbers
    """
    
    if len(args)==0: return 1

    greater=max(args)
    maxsearch=1
    for arg in args: maxsearch*=arg
    while(greater <=maxsearch):
        
        if all(greater % int(val) ==0 for val in args):
            return greater
            break
        
        greater+=1

def circularTimeSlides(detectors,Nstep):
    """This function create all the possible circular lags for a given
    number of detectors and a given number of segments available.
    
    
    Parameters
    ----------
    
    detectors : `int`
        The number of detectors
        
    Nstep : `int`
        The number segments we want to calculate the circular lags for.
       
    Returns
    -------
    timeSlideTable: `list` [`list`]
        A list of lists, in which it presents all the independent lags possible
        given the input parameters. Each line (list element) is a different 
        rearangement of the lags.
        
    Note
    ----
    This function returns the time slide table, or lag table, but this is 
    independent of the duration the sements that are lagged have. If your
    rearangement is of time **T** you need to multiply the table by that time.
    
    If there are not enough segments to do the lags, only the zero-lag will
    be returned.
    
    Examples
    --------
    If we have 3 detectors the lowest common multiple of 1:detectors-1
    is the lowest common multiple of [1,2] which is 2. Them we look for the 
    largest multiple of two [2,4,6,...] X such that orginal Nstep is equal or
    bigger than X+1. If we have a segment with 128 seconds and we look 
    for 1s intervals then original Nstep=128, so the multiple we look for is 
    X=126, so X+1=127 < 128. If we have 4 detectors the lowest common multiple 
    is 6. Then the largest multiple of 6 that is smaller than 128 is 126 (21*6).
    
    References
    ----------
    This code is based on circulartimeslides.m 5956 2021-03-30 11:52:17Z 
    patrick.sutton@LIGO.ORG in x-pipeline from Patrick Sutton.
    """
    
    if not(isinstance(detectors,int)) and len(detectors)>=1:
        detectors=len(detectors)

    if not ((isinstance(detectors,int) and detectors>=1)):
        raise TypeError('Number of detectors must be a natural number >= 1.')
    Ndet = detectors

    if not(isinstance(Nstep,int) or Nstep>1):
        raise ValueError('Input Nstep must be a natural number.')


    # ---- Calculation depends on number of detectors.
    if Ndet==1:
        c = [[0]];
    else:

        LCM = lcm(*tuple(range(1,Ndet)))

        Nstep = int((Nstep-1)/LCM)*LCM+1

        # ---- Circular time steps.
        stepArray=np.arange(0,Nstep)
        detArray=np.arange(0,Ndet)
        c = np.mod(np.dot(stepArray.reshape(len(stepArray),1)
                          ,detArray.reshape(1,len(detArray))),Nstep)
    return c


def internalLags(detectors             # The initials of the detectors you are going 
                 ,duration             # The duration of the instances you use
                 ,size                 # Size in seconds of the available segment
                 ,fs=None              # Sample frequency
                 ,start_from_sec=None  # The first second in the segment
                 ,lags=None            # Time slides multiples returned
                 ,includeZeroLag=True):# Includes zero lag by defult !!!    
    
    '''This function uses circularTimeLags to create a lag table and then
    applies the time options and sample frequency given to create indeces 
    that correspod to them.
    
    Parameters
    ----------
    detectors : `str` 
        It can be any combination of the intials of detectors  
        (without duplication), {H,L,V,K,I,U} and U stands for undefined.
        So the parameter should be 'HLV' for three detector network.
    duration : `int` or `float`
        The duration of the segment time-lagged in seconds
    size : `int`
        The size in seconds of the available segments to be time-lagged.
    fs : `int`
        The sample frequency, defaults to 1.
    start_from_sec : `int` or `float`
        Positive constant to start the indeces from that constant (seconds).
        Usually used to ignore first seconds
    lags : `int`
        The amount of circularTimeSlides table lines we want to be returned.
        Usually we don't need all of them. Defaults to 1.
    includeZeroLag : `bool`
        Option if we want to include zero-laged segments in the data.
    '''
    
    if fs==None: fs=1
    if start_from_sec==None: start_from_sec=0
    if lags==None: lags=int(size/duration)
        
    # The following function creates all the possible shifts for the available
    # detectors, size and duration
    C=circularTimeSlides(detectors,int(size/duration))
    # Removing the zero lag
    if includeZeroLag==False:
        C=C[1:]  
    C=C[:lags+1]
    # Creation of indeces
    IND={}
    # The indeces are returned in a dictionary format for each detector
    for d in detectors:
        IND[d]=[]
    
    # Creation of indeces for all available lags.
    for lag in C:
        for i in range(len(detectors)):
            IND[detectors[i]]+=np.roll(# The same array is generated all the time and
                    start_from_sec*fs  # the starting second is addedd
                    +np.arange(0,int(size/duration)*duration*fs,duration*fs)
                    ,lag[i]).tolist() # and the shift from the circular lags is applied
            
    return(IND)


    
    
def toCategorical(labels,translation = True,from_mapping=None):
    """This function translate a list of labels to a categorical format. It also
    saves a dictionary translation of what label corresponds to what category. 
    
    Parameters
    ----------
    labels : `list` [`str`]
        A list of strings with the labels that you want to turn to categorical.
    translation : `bool`
        Option to return the reverse translation
    from_mapping : `list` [`str`] , optional
        The option to define the order that the pairs of label 
        category and categorical format are assigned. For example,
        by passing this mapping: ['signal', 'noise'] we make sure that
        the first label will go to 'signal' and the second to
        'noise' : {'signal':[1,0], 'noise'[0,1]}.
    
    Returns
    -------
    categorical list : `list`
        The list of categorical labels if translation == False
    categorical list, translation dictionary : `list`, `dict`
        The list of categorical labels and the translation dictionary of 
        those labels, if translation == True
        
    Note
    ----
    It is advised to use the from_mapping option, because when the code
    chooses which label goes to what category, it assigs them based on 
    the order they appear. If you apply this function to two different
    datasets with different order of labels you will have a different 
    translation.
        
    
    """
    # If mapping is not given we use the labels in the same
    # order they appear in their list.
    if from_mapping == None:
        classes=[]
        classesNumerical=[]
        num=0
        for label in labels:
            if label not in classes: 
                classes.append(label)
                classesNumerical.append(num)
                num+=1
                
    # If mapping is given, we check if the all the strings in the 
    # labels argument are present to the mapping.
    elif isinstance(from_mapping,list):
        classes=from_mapping
        for label in labels:
            if label not in classes: 
                raise ValueError(str(label)+' was not included in your mapping')
        classesNumerical=list(i for i in range(len(classes)))
    
    else:
        raise TypeError("from_mapping must be a list of the label instances"+
                        " with your prefered order.")

    # To this point we have defined the classes and and their numbers assigned
    # to them.
    translationDict={}
    classesCategorical = []
    
    # Creation of categorical classes and their translation.
    for i in range(len(classes)):
        categorical = len(classes)*[0]
        categorical[i]=1
        classesCategorical.append(categorical)
        translationDict[classes[i]] = categorical
        
    # Creating the list of labels with the categorical format.
    labelsCategorical=[]
    
    for label in labels:
        labelsCategorical.append(translationDict[label])
    labelsCategorical=np.asarray(labelsCategorical)
    
    # Returning categorical classes and translation if required.
    if translation==True:
        return(labelsCategorical,translationDict)
    else:
        return(labelsCategorical)
     
    
    
def correlate(x,y,window):
    """Function that calculates the Pearson correlation of two arrays x and y 
    for a given window of pixels
    
    Parameters
    ----------
    x : `nd.array` or `list` [`float`], (N,)
        The first timeseries
    y : `nd.array` or `list` [`float`], (N,)
        The second timeseries    
    window : `int`
        The window of pixels to to which we calculate the correlation.
        We shift timeseries y in respect to x from -window to window.
    
    Returns
    -------
    correlation : `numpy.ndarray` (2*window,)
        The correlation timeseries.
    """
    
    if len(x)!=len(y):
        raise ValueError("Arrays have to be of the same length.")
    result=[]
    for i in np.arange(-window,window):
        xroll=np.roll(x,i)
        cor = np.sum((xroll-np.mean(x))*(y-np.mean(y)))/(
            np.sqrt(np.sum((xroll-np.mean(x))**2))*np.sqrt(np.sum((y-np.mean(y))**2)))
        result.append(cor)
    return np.array(result)




    
def externalLags(detectors
                   ,duration
                   ,size):
    """This function acts as a wrapper function on internalLags
    to calculate externalLags
    
    """
    size=int(size/duration)
    ind=internalLags(detectors
                       ,duration
                       ,size
                       ,includeZeroLag=False)
    
    return(ind)





def fromCategorical(label,mapping,column=False):
    """The reverse function from the toCategorical. It uses a
    translation mapping to translate the label back to its
    original string format.
    
    Parameters
    ----------
    label : `list` [`int`]
        The categorical forma of the label.
    mapping : `dict`
        The mapping dictionary.
    column : `bool` , optional
        If True, it returns the index
        
    Returns
    -------
    label : `str`
        The label in its original string format.

    """
    if isinstance(mapping,dict):
        dictionary = mapping
    else:
        raise TypeError('You need to provide mapping in dictionary form.')
    categorical = list(dictionary.values())
    labels = list(dictionary.keys())

    if column == True:
        return(np.where(np.array(dictionary[label])==1)[0][0])
    else:
        return(dictionary[label]) 


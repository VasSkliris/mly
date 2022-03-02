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

# This function creates a list of indeses as begining for every instantiation of
# noise generated. This makes the lag method happen.

# def index_combinations(detectors
#                        ,lags
#                        ,length
#                        ,fs
#                        ,size
#                        ,start_from_sec=0):
    
#     indexes={}

#     if lags==1:
#         for det in detectors:
#             indexes[det]=np.arange(start_from_sec*fs,start_from_sec*fs
#                                    +size*length*fs, length*fs)
            
#     elif lags>=len(detectors):
        
#         batches=int(ceil(size/(lags*(lags-1))))
        
#         for det in detectors:
#             indexes[det]=np.zeros(lags*(lags-1)*batches)

#         d=np.int_(range(lags))

#         if len(detectors)==1:
#             indexes[detectors[0]]=np.arange(start_from_sec*length*fs
#                        ,(start_from_sec+size)*length*fs, length*fs)



#         elif len(detectors)==2:
#             for b in range(0,batches):
#                 for j in range(1,lags):
#                     indexes[detectors[0]][(b*(lags-1)+j-1)*lags
#                     :(b*(lags-1)+j)*lags]=(start_from_sec+(b*lags+d)*length)*fs
                    
#                     indexes[detectors[1]][(b*(lags-1)+j-1)*lags
#                     :(b*(lags-1)+j)*lags]=(start_from_sec
#                                            +(b*lags+np.roll(d,j))*length)*fs
                    
#         elif len(detectors)==3:
#             for b in range(0,batches):
#                 for j in range(1,lags):

#                     indexes[detectors[0]][(b*(lags-1)+j-1)*lags
#                     :(b*(lags-1)+j)*lags]=(start_from_sec+(b*lags+d)*length)*fs
                    
#                     indexes[detectors[1]][(b*(lags-1)+j-1)*lags
#                     :(b*(lags-1)+j)*lags]=(start_from_sec
#                                            +(b*lags+np.roll(d,j))*length)*fs
#                     indexes[detectors[2]][(b*(lags-1)+j-1)*lags:(b*(lags
#                                         -1)+j)*lags]=(start_from_sec
#                                         +(b*lags+np.roll(d,-j))*length)*fs

#     for det in detectors:
#         indexes[det]=np.int_(indexes[det][:size])

#     return(indexes)
    



# def get_ligo_noise(days
#                    , fs=1024 
#                    , detectors='HLV' 
#                    , destinationPath=None 
#                    , minDuration=None
#                    , key=None):
    
#     """This functions will download all the strain ligo data that coincide 
#     between all detectors specified for the desired days. Furthermore it will
#     downsample them to the specfied sample frequency (fs). More specifically,
#     for a given day it will create a directory at the destinationPath, this 
#     directory will have a short version of the date as a name ex. 12 August 2017 
#     --> 20170812. Inside this directory, a directory for each detector will be created.
#     For every day the function checks when these channels H1:DMT-ANALYSIS_READY:1
#     , L1:DMT-ANALYSIS_READY:1, V': 'V1:ITF_SCIENCE:1' or some of these chennels given
#     the detectors option. The segments that are overlapping are downloaded (SEG#). Additionally 
#     all segments are checked for zero gaps and in that case the are fragmenteted even
#     more with a letter as an indicatior(SEG0 --> SEG0, SEG0b). Finally for every segment
#     the beggining gps time and the duration of the segment in seconds is writen on the name 
#     creating files like SEG0c_122345678_5555.txt which have the same name at all 
#     detector directories.

#     Parameters
#     ----------
    
#     days: list of strings
#         This argument is a list of days of which data we want to download. 
#         Days must be in the date format ex: ['13 July 2017', ... ]
        
#     fs: int
#         The sample frequency of the final data. If not specified it set to 1024.
    
#     detectors: string/list of strings 
#         The detectors that we care to have overlaping times. If not specified
#         all detectors ('HLV') option is set.
    
#     destinationPath: string/ path (optional)
#         The desired path for the data to be structured. If not specified it is
#         set to the current directory.
    
#     minDuration: int (optional)
#         The minimum duration of segments we want. Sometimes for example having
#         a segment of 3 seconds might create problems. If not specified it is set
#         to 128.
    
#     key: str path (optional)
#         If you need permision to download the data you can use a kerberos key.
#         If needed specify the path to the key here.
    
#     Notes
#     -----
    
#     This function downloads the following frametypes{'H': 'H1_HOFT_C00'
#     ,'L': 'L1_HOFT_C00','V': 'V1Online'}
#     """
    
#     if isinstance(days,str): days = [days]
#     if not all(isinstance(d,str) for d in days): 
#         raise ValueError('Days must be in the date format ex: 13 July 2017')
        
#     if not (isinstance(fs,int) and fs%2==0):
#         raise ValueError('Sample frequency must be a valid number miltpiple of 2')
        
#     if not ((isinstance(detectors,str) and (det in 'HLV' for det in detectors))
#              or (isinstance(detectors,list)) and (det in 'HLV' for det in detectors)):
#         raise ValueError("Detectors can be any collection of 'H','L','V' in string or list format")
        
#     if destinationPath==None:
#         destinationPath='./'
        
#     if minDuration==None:
#         minDuration = 128
        
#     if isinstance(key,str):
#         kinit(keytab = key)
        
#     for day in days:
        
        
#         gps_start = to_gps(day)
#         gps_end = gps_start+24*3600     
#         GPS = [gps_start, gps_end]
        
#         # Looking for segments that data exists

#         det_seg_channels = {'H': 'H1:DMT-ANALYSIS_READY:1'
#                            ,'L': 'L1:DMT-ANALYSIS_READY:1'
#                            ,'V': 'V1:ITF_SCIENCE:1'}
#         det_segs=[]
#         for det in detectors:
#             det_segs.append(query_segments(det_seg_channels[det], GPS[0],GPS[1]))

#         # Finding the time segments where all detectors are on observing mode.
        
#         sim_seg = det_segs[0]['active']
#         for seg in det_segs[1:]:
#             sim_seg = sim_seg & seg['active']

#         if not sim_seg==[]:

#             seg_time=[]      
#             sim_time_total=0 
#             seg_no=0         

#             maxseg=10000 # We don't want huge segments
#             for seg in sim_seg:
#                 if seg[1]-seg[0] >maxseg:
#                     for r in range(int((seg[1]-seg[0])//maxseg)):
#                         seg_time.append([seg[0]+r*maxseg,seg[0]+(r+1)*maxseg])
#                         sim_time_total+=maxseg
#                         seg_no+=1
#                     seg_time.append([seg[0]+maxseg*((seg[1]-seg[0])//maxseg),seg[1]])
#                     sim_time_total+=(seg[1]-(seg[0]+maxseg*((seg[1]-seg[0])//maxseg)))
#                     seg_no+=1
#                 elif seg[1]-seg[0] < minDuration:
#                     continue
#                 else:
#                     seg_time.append([seg[0],seg[1]])
#                     sim_time_total+=(seg[1]-seg[0])
#                     seg_no+=1
#             print('There are '+str(seg_no)
#                   +' segments of simultaneous GPS time, with total duration of \n'
#                   +str(sim_time_total)+' seconds  =  '+str(sim_time_total/3600)+' hours')
#             print('More specifically:')
#             for i in range(0,len(seg_time)):
#                 print('Segment No:'+str(i)+' Time: '+str(seg_time[i][0])
#                       +' to '+str(seg_time[i][1])+' Duration: '+str(seg_time[i][1]-seg_time[i][0]))    

#             # Creating director name acording to date

#             year=str(from_gps(gps_start).year)
#             month=str(from_gps(gps_start).month)
#             day_=str(from_gps(gps_start).day)

#             if len(month)==1: month='0'+month
#             if len(day_)==1: day_='0'+day_

#             filename = year + month + day_

#             if os.path.isdir(destinationPath+filename):
#                 pass
#                 # avoiding overighting
#                 #raise NameError('Directory '+filename+' already exists')
#             else:
#                 os.mkdir(destinationPath+filename)
#                 for det in detectors:
#                     os.mkdir(destinationPath+filename+'/'+det)
#             print('File '+destinationPath+filename+' has been created.')


#             # Getting the data and saving them downsampled to the desired sample frequency

#             det_channels = {'H': 'H1:GDS-CALIB_STRAIN'
#                            ,'L': 'L1:GDS-CALIB_STRAIN'
#                            ,'V': 'V1:Hrec_hoft_16384Hz'}
#             det_frametypes = {'H': 'H1_HOFT_C00'
#                              ,'L': 'L1_HOFT_C00'
#                              ,'V': 'V1Online'}
#             for segment in range(seg_no):

#                 gpsStart = seg_time[segment][0]
#                 gpsEnd   = seg_time[segment][1]
#                 data_dict={}
#                 nameref_dict={}
#                 if gpsEnd-gpsStart >= minDuration:
                    
#                     for det in detectors:

#                         try:
#                             data=TimeSeries.get( det_channels[det]
#                                                  , gpsStart
#                                                  , gpsEnd
#                                                  , frametype = det_frametypes[det]
#                                                  , verbose=True).astype('float64')
#                         except: # using the kerberos key if permition is expired
#                             if isinstance(key,str):
#                                 os.system("ligo-proxy-init -k")
#                                 try:
#                                     data=TimeSeries.get( det_channels[det]
#                                                          , gpsStart
#                                                          , gpsEnd
#                                                          , frametype = det_frametypes[det]
#                                                          , verbose=True).astype('float64')
#                                 except:
#                                     break
                                    
#                             else:
#                                 break
                                
#                         data_dict[det]=data.resample(fs) # resampling the data
#                         nameref_dict[det] = str('SEG'+str(segment)+'_'+str(int(gpsStart))
#                                       +'_'+str(int(gpsEnd-gpsStart))+'s.txt')
                        
#                     # In case one of the detectors fails to give data we ignore the segment.
#                     if len(detectors)!=len(list(data_dict.keys())): continue

#                     # Checking for zeros is a complicated procedure.
#                     # We check every second insted of each element to make it faster.
#                     intervalList=[]
#                     for det in detectors:
#                         strain=data_dict[det][::fs]
#                         non_zero_indexes=np.where(strain != 0)[0]
#                         # The intervals are the intervals with non zero data.
#                         intervals=SegmentList([])
#                         interval=[]
#                         if strain[0]!=0 : interval.append(0)
#                         for i in range(1,len(strain)):
#                             if (strain[i]==0 and strain[i-1]!=0 and len(interval)==1):
#                                 if (i-1-interval[0])>=128:
#                                     interval.append(i-1)
#                                     print(interval)
#                                     intervals.append(Segment((np.array(interval)*fs).tolist()))
#                                     interval=[]
#                                 else:
#                                     interval=[]
#                             elif (strain[i]!=0 and strain[i-1]==0 and len(interval)==0):
#                                 interval.append(i)

#                             if i==len(strain)-1 and len(interval)==1:
#                                 if ((len(strain)-1-interval[0])>=128):
#                                     interval.append(i-1)
#                                     print(interval)
#                                     intervals.append(Segment((np.array(interval)*fs).tolist()))
#                                     interval=[]
#                                 else:
#                                     interval=[]
#                         print(intervals)
#                         if len(strain)-len(non_zero_indexes) == 0:
#                             print('   '+det+' - '+nameref_dict[det]+' has no zeros')
#                         else:
#                             print('   '+det+' - '+nameref_dict[det]+' had zeros '+str(np. array(intervals)*fs))

#                         intervalList.append(intervals)
#                     intervRef=intervalList[0]
#                     for interv in intervalList: 
#                         intervRef=intervRef & interv
#                         print(intervRef)

#                     intervals=list(list(s) for s in intervRef.coalesce())
#                     print(intervals)
#                     if len(intervals)!=0:
#                         sublabels=['','b','c','d','e','f','g','h','i','j','k']
#                         for d in detectors:
#                             for I in range(len(intervals)): 
#                                 sub_duration=int((intervals[I][1]-intervals[I][0])/fs)
#                                 sub_strain=data_dict[d][intervals[I][0]:intervals[I][0]+sub_duration*fs]
#                                 sub_seg=nameref_dict[d].split('_')[0]+sublabels[I]+'_'+str(int(nameref_dict[d].split('_')[1])+int(intervals[I][0]/fs))+'_'+str(sub_duration)+'.txt'
#                                 print(destinationPath+filename+'/'+d+'/'+sub_seg+' has been created')

#                                 np.savetxt(destinationPath+filename+'/'+d+'/'+sub_seg,sub_strain)
#                                 print('File .../'+d+'/'+sub_seg+' is successfully saved')

#                     # end check of zeros


#                 else:
#                     print('Duration '+str(int(gpsEnd-gpsStart))+'s is below minimum duration')

#         else:
#             print('No simultaneous GPS time for '+day+', sorry ...')

            




# def old_get_ligo_noise(days
#                    , fs=1024 
#                    , detectors='HLV' 
#                    , destinationPath=None 
#                    , minDuration=None
#                    , key=None):
    
#     # LOG IN WITH LIGO CREDENTIALS
#     #if key=='vs' : kinit(keytab= '/home/vasileios.skliris.keytab' )
#     #!ligo-proxy-init -k
    
#     if isinstance(days,str): days = [days]
#     if not all(isinstance(d,str) for d in days): 
#         raise ValueError('Days must be in the date format ex: 13 July 2017')
        
#     if not (isinstance(fs,int) and fs%2==0):
#         raise ValueError('Sample frequency must be a valid number miltpiple of 2')
        
#     if not ((isinstance(detectors,str) and (det in 'HLV' for det in detectors))
#              or (isinstance(detectors,list)) and (det in 'HLV' for det in detectors)):
#         raise ValueError("Detectors can be any collection of 'H','L','V' in string or list format")
        
#     if destinationPath==None:
#         destinationPath='./'
        
#     if minDuration==None:
#         minDuration = 128
        
#     if isinstance(key,str):
#         kinit(keytab = key)
#         os.system("ligo-proxy-init -k")
        
#     for day in days:
        
        
#         gps_start = to_gps(day)
#         gps_end = gps_start+24*3600     
#         GPS = [gps_start, gps_end]
        
#         # Looking for segments that data exists

#         det_seg_channels = {'H': 'H1:DMT-ANALYSIS_READY:1'
#                            ,'L': 'L1:DMT-ANALYSIS_READY:1'
#                            ,'V': 'V1:ITF_SCIENCE:1'}
#         det_segs=[]
#         for det in detectors:
#             det_segs.append(query_segments(det_seg_channels[det], GPS[0],GPS[1]))

#         # Finding the time segments where all detectors are on observing mode.
        
#         sim_seg = det_segs[0]['active']
#         for seg in det_segs[1:]:
#             sim_seg = sim_seg & seg['active']

#         if not sim_seg==[]:

#             seg_time=[]      
#             sim_time_total=0 
#             seg_no=0         

#             maxseg=10000
#             for seg in sim_seg:
#                 if seg[1]-seg[0] >maxseg:
#                     for r in range(int((seg[1]-seg[0])//maxseg)):
#                         seg_time.append([seg[0]+r*maxseg,seg[0]+(r+1)*maxseg])
#                         sim_time_total+=maxseg
#                         seg_no+=1
#                     seg_time.append([seg[0]+maxseg*((seg[1]-seg[0])//maxseg),seg[1]])
#                     sim_time_total+=(seg[1]-(seg[0]+maxseg*((seg[1]-seg[0])//maxseg)))
#                     seg_no+=1
#                 elif seg[1]-seg[0] < minDuration:
#                     continue
#                 else:
#                     seg_time.append([seg[0],seg[1]])
#                     sim_time_total+=(seg[1]-seg[0])
#                     seg_no+=1
#             print('There are '+str(seg_no)+' segments of simultaneous GPS time, with total duration of \n'
#                   +str(sim_time_total)+' seconds  =  '+str(sim_time_total/3600)+' hours')
#             print('More specifically:')
#             for i in range(0,len(seg_time)):
#                 print('Segment No:'+str(i)+' Time: '+str(seg_time[i][0])
#                       +' to '+str(seg_time[i][1])+' Duration: '+str(seg_time[i][1]-seg_time[i][0]))    

#             # Creating director name acording to date

#             year=str(from_gps(gps_start).year)
#             month=str(from_gps(gps_start).month)
#             day_=str(from_gps(gps_start).day)

#             if len(month)==1: month='0'+month
#             if len(day_)==1: day_='0'+day_

#             filename = year + month + day_

#             if os.path.isdir(destinationPath+filename):
#                 pass
#                 #raise NameError('Directory '+filename+' already exists')
#             else:
#                 os.mkdir(destinationPath+filename)
#                 for det in detectors:
#                     os.mkdir(destinationPath+filename+'/'+det)
#             print('File '+destinationPath+filename+' has been created.')


#             # Getting the data and saving them downsampled to the desired sample frequency

#             det_channels = {'H': 'H1:GDS-CALIB_STRAIN'
#                            ,'L': 'L1:GDS-CALIB_STRAIN'
#                            ,'V': 'V1:Hrec_hoft_16384Hz'}
#             det_frametypes = {'H': 'H1_HOFT_C00'
#                              ,'L': 'L1_HOFT_C00'
#                              ,'V': 'V1Online'}
#             for segment in range(seg_no):

#                 gpsStart = seg_time[segment][0]
#                 gpsEnd   = seg_time[segment][1]
                
#                 if gpsEnd-gpsStart >= minDuration:

#                     for det in detectors:
                        
#                         try:
#                             data=TimeSeries.get( det_channels[det]
#                                                  , gpsStart
#                                                  , gpsEnd
#                                                  , frametype = det_frametypes[det]
#                                                  , verbose=True).astype('float64')
#                         except:
#                             if isinstance(key,str):
#                                 os.system("ligo-proxy-init -k")
#                                 try:
#                                     data=TimeSeries.get( det_channels[det]
#                                                          , gpsStart
#                                                          , gpsEnd
#                                                          , frametype = det_frametypes[det]
#                                                          , verbose=True).astype('float64')
#                                 except:
#                                     break
#                             else:
#                                 break
#                         t0=time.time()
#                         print(len(data)/(16*fs))
#                         data=data.resample(fs)
#                         t1=time.time()-t0
#                         print(t1)

#                         nameref = str('SEG'+str(segment)+'_'+str(int(gpsStart))
#                                       +'_'+str(int(gpsEnd-gpsStart))+'s.txt')

#                         np.savetxt(destinationPath+filename+'/'+det+'/'+nameref,data)
#                         print('File '+nameref+' is successfully saved')


#                 else:
#                     print('Duration '+str(int(gpsEnd-gpsStart))+'s is below minimum duration')
                                              
                    
#         else:
#             print('No simultaneous GPS time for '+day+', sorry ...')



            


# def get_ligo_glitches(dayStart                   
#                    , outputDuration
#                    , dayEnd=None
#                    , fs=1024 
#                    , destinationPath=None 
#                    , key=None
#                    , **kwargs):
    
#     # LOG IN WITH LIGO CREDENTIALS
#     #if key=='vs' : kinit(keytab= '/home/vasileios.skliris.keytab' )
#     #!ligo-proxy-init -k
    
#     detectors=['H1','L1','V1']
#     runs={ 'O1':[1126623617,1136649617] 
#           ,'O2':[1164556817,1187733618]
#           ,'O3a':[1238166018,1253977218]
#           ,'O3b':[1256655618,1269363618]}
    
#     if dayStart in list(runs.keys()):
#         run=dayStart
#         theRun=dayStart
#         dayStart=runs[run][0]
#         dayEnd=runs[run][1]
#     else: 
#         theRun=None

        
#     if isinstance(dayStart,str): dayStart=to_gps(dayStart)
        
#     if dayEnd==None: dayEnd=dayStart+3600*24
#     if isinstance(dayEnd,str): dayEnd=to_gps(dayEnd)
        
                
#     if not (isinstance(fs,int) and fs%2==0):
#         raise ValueError('Sample frequency must be a valid number miltpiple of 2')
        
#     if destinationPath==None:
#         destinationPath='./'

#     if isinstance(key,str):
#         kinit(keytab= key)
#         os.system("ligo-proxy-init -k")

#     # load glitch dataFrame
#     print(dayStart,dayEnd)
#     for run in list(runs.keys()):
#         if (runs[run][0]<=dayStart<=runs[run][1] 
#             and runs[run][0]<=dayEnd<=runs[run][1]):
#             theRun=run
#     if theRun==None:
#         raise ValueError('You can only get glitches in one Run at a time.\n'
#                              +'dayStart and dayEnd are from different runs')
    
#     glitchFrame=pd.read_csv('glitch_'+theRun+'_H1.csv')
#     for d in range(1,3):
#         glD=pd.read_csv('glitch_'+theRun+'_'+detectors[d]+'.csv')
#         glitchFrame=glitchFrame.append(glD,ignore_index=True)
    
#     glitchFrame=glitchFrame[glitchFrame['GPStime']>=dayStart]
#     glitchFrame=glitchFrame[glitchFrame['GPStime']<=dayEnd]
    
#     glitchFrame['maxFreq']=glitchFrame['centralFreq']+glitchFrame['bandwidth']/2

#     print(kwargs)
#     for option in kwargs:
#         if option not in list(glitchFrame.columns):
#             raise KeyError(option+' is not a valid option')
#         if (isinstance(kwargs[option],tuple)
#             and len(kwargs[option])==2 
#             and all(isinstance(op,(int,float)) for op in kwargs[option])):
#             glitchFrame=glitchFrame[kwargs[option][0]<=glitchFrame[option]]
#             glitchFrame=glitchFrame[kwargs[option][1]>=glitchFrame[option]]
#         elif isinstance(kwargs[option],list):
#             glitchFrame=glitchFrame[glitchFrame[option] in kwargs[option]]
        
#         elif isinstance(kwargs[option],(str,int,float)):
#             glitchFrame=glitchFrame[glitchFrame[option]==kwargs[option]]
        

    
#     glitchFrame=glitchFrame.sort_values(by='GPStime')
#     gl=glitchFrame.reset_index(drop=True)
    
#     window=16*outputDuration

#     det_channels = {'H1': 'H1:GDS-CALIB_STRAIN'
#                    ,'L1': 'L1:GDS-CALIB_STRAIN'
#                    ,'V1': 'V1:Hrec_hoft_16384Hz'}
#     det_frametypes = {'H1': 'H1_HOFT_C00'
#                      ,'L1': 'L1_HOFT_C00'
#                      ,'V1': 'V1Online'}
    
#     gl['filename']=gl.shape[0]*[None]
#     for i in range(gl.shape[0]):
        
#         ifo=gl['ifo'].iloc[i]
#         GPS=[ gl.loc[[i],['GPStime']].values[0][0]-window/2
#              ,gl.loc[[i],['GPStime']].values[0][0]+window/2]    
        
#         fileName=gl.loc[[i],['id']].values[0][0]+'_'+str(int(gl.loc[[i],['GPStime']].values[0][0]))+'_1s.txt'
        
#         if os.path.isfile(destinationPath+fileName):
#             gl.loc[[i],['filename']]= fileName
#             continue
#         try:
#             data=TimeSeries.get(det_channels[ifo]
#                                 , GPS[0],GPS[1]
#                                 , frametype=det_frametypes[ifo]
#                                 ,verbose=False).astype('float64')
#             data=data.resample(fs)
#             print(len(data))
#             D=np.array(data.whiten(1,0.5)[int(fs*(window-outputDuration)/2): int(fs*(window+outputDuration)/2)])
#             D=D-np.mean(D)
#             np.savetxt(destinationPath+fileName,D)
#             gl.loc[[i],['filename']]= fileName
            
#             print(len(D))
#         except:
#             try:
#                 os.system("ligo-proxy-init -k")
#                 data=TimeSeries.get(det_channels[ifo]
#                                     , GPS[0],GPS[1]
#                                     , frametype=det_frametypes[ifo]
#                                     ,verbose=False).astype('float64')
#                 data=data.resample(fs)
#                 print(len(data))
#                 D=np.array(data.whiten(1,0.5)[int(fs*(window-outputDuration)/2): int(fs*(window+outputDuration)/2)])
#                 D=D-np.mean(D)
#                 gl.loc[[i],['filename']]= fileName
#                 np.savetxt(destinationPath+fileName,D)
#                 print(len(D))
#             except:
#                 print('skipped gitch')
#                 gl.loc[[i],['filename']]= None
        
#     gl.to_pickle(destinationPath+'index.pkl')

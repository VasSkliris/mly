from ..simulateddetectornoise import *  
from ..tools import dirlist, internalLags, correlate
from ..plugins import *

from .datatools import DataSet, DataPod

import os
import time
import gwdatafind
import pickle
import random
import subprocess


import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from math import ceil
from pycondor import Job, Dagman

from matplotlib.mlab import psd
from gwpy.time import to_gps, from_gps
from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment,SegmentList,DataQualityFlag
from dqsegdb2.query import query_segments
from scipy.stats import pearsonr
from scipy.special import comb
from urllib.error import HTTPError

which_python = subprocess.check_output('which python', shell=True, text=True)


def injection_initialization(injection_source, detectors):

    '''Processing of the injection format in the generator

    '''

    if injection_source == None:
        inj_type = None

    # Path input
    elif isinstance(injection_source, str): 

        # directory with pods path
        if os.path.isdir(injection_source):

            if injection_source[-1] != "/": injection_source+="/" 

            if all(os.path.isdir(injection_source+det) for det in detectors):

                inj_type = 'oldtxt'
            else:
                inj_type = 'directory'

        # dataPod or dataSet path
        elif (os.path.isfile(injection_source) and injection_source[-4:]=='.pkl'):
            with open(injection_source,'rb') as obj:
                injection_source = pickle.load(obj)
            
            if isinstance(injection_source,DataPod):
                inj_type = 'DataPod'

            elif isinstance(injection_source,DataSet):
                if len(injection_source) > 0:
                    inj_type = 'DataSet'
                else:
                    raise ValueError("injection_source DataSet is empty")

        else:
            raise FileNotFoundError('Not valid directory for :'+injection_source)
        
    # DataSet                     
    elif isinstance(injection_source,DataSet):

        if len(injection_source) > 0:
            inj_type = 'DataSet'
        else:
            raise ValueError("injection_source DataSet is empty")

    # DataPod
    elif isinstance(injection_source,DataPod):
        inj_type = 'DataPod'

    else:
        raise TypeError('Not valid input for injection_source: ',injection_source 
                    ,"\nIt has to be either a folder or a DataPod or DataSet object.")

    return injection_source, inj_type





def generator(duration
               ,fs
               ,size
               ,detectors
               ,injection_source = None
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
               ,frames=None 
               ,channels=None
               ,disposition=None
               ,maxDuration=None
               ,differentSignals=False   # In case we want to put different injection to every detector.
               ,plugins=None
               ,**kwargs):

    """This is one of the core methods/function of this module.
    It generates data instances of detector strain data that could be 
    from real detectors or simulated noise. It has many options for
    each different type of instance -that it can generate.

    Parameters
    ----------

    duration : float/int
        The duration of instances to be generated in seconds.

    fs : float/ing
        The sample frequency of the instances to be generated.

    size : int
        The number of instances to be generated.

    detectors : str/ list of str
        The detectors to be used for the background. Even for optimal 
        gaussian noise the background is different for each detector. 
        For example if you wan to include Hanford, Livingston and Virgo
        you state it as 'HLV' or ['H','L','V'].

    injection_source: str (path, optional)
        The path to the directory with the injections. If not specified
        the generator will just generate noise instances, setting
        injectionSNR to 0.

    labels: dict (optional)
        The optional label for all the instances to be generated. 
        If not specified labels is {'type':'UNDEFINED'}.

    backgroundType: {'optimal','real'}
        The type of background to be used. If 'optimal' is chosen,
        a simulated noise background will be used (simulateddetectornoise.py)
        depending the detector. If 'real' is chosen, a source of real detector
        noise will need to be specified from noiseSourceFile.

    injectionSNR: float/int/list/tuple/callable
        The Signal to Noise Ration of the injection used for the instances.
        All injections are calibrated to be in that SNR value. A range can 
        be used as also a function with no parameters tha just generates values.
        This is usefull for distributions of events.

    noiseSourceFile: {gps intervals, path, txt file, numpy.ndarray}
        The source of real noise. There are many options to provide source 
        of real noise. 

        * A list intervals (one for each detectors) with gps times. The script 
        will get the available coinsident data within the 'detectors' and use them.

        * A path with txt or DataPod files that have noise data.

        * A numpy.ndarray with the data directly.

        This parameter evolves depending on the needs of users.

    windowSize: int/float
        The amount of seconds to use around the instance duration, for the 
        calculation of PSD. The bigger the windowSize the better whittening.
        Although the calculation takes more time. It defaults to 16 times
        the duration parameter.

    timeSlides: int (optional)
        The number of timeshifts to use on the provided noise source.
        If noise source has N instances of utilisable noise by using
        timeSlides we will have N*(1-timeSlides) instances. Default
        is 1 which means not using timeSlides.

    startingPoint: float/int (optional)
        The starting point in seconds from which we will use the noise
        source. In case we have a big noise sample by specifing startingPoint
        we do not start from the beggining but from that second. Default is 0.

    name : str (optional)
        The name of the dataset, if it is saved. 

    savePath : str (path, optional)
        The of the file to be saved. If not stated file will not be saved and
        it will just be returned.

    single : bool
        If true and injectionSNR or injectionHRSS is defined, it will add
        a signal to only one of the available detectors, randomly selected.
        All SNR or hrss will be used on the single injection. Default is False.

    differentSignals : bool
        If true each detector gets a different randomly selected signal. 
        Default is False.

    injectionCrop : float [0,1]
        Allows to crop part of the injection (up to the value provided when 
        randomly positions the injection arroud on the background. 0 means
        no croppin allowd and 1 means maximum 100% cropping allowed. The cropping 
        will be a random displacement from zero to parto of durationn.

    frames: dict or {C00, C01,C02}
        In case we use real detector noise and the script looks for the available 
        noise segments we need to specify the frames to look for. We have to provide 
        a dictionary of the frames for the given detectors. C00, C01 and C02 provides a
        shortcut for the widly used frames. 
        {'H': 'H1_HOFT_C0X' ,'L': 'L1_HOFT_C0X' ,'V': 'V1Online'}.
        C02 is chossen as default given that is pressent in all observing runs.

    channels: dict or {C00, C01,C02} 
        As with frames, in case we use real detector noise and the script looks for 
        the available noise segments we need to specify the channels to look for. 
        We have to provide a dictionary of the channels for the given detectors. 
        C00, C01 and C02 provides a shortcut for the widly used frames. 
        {'H': 'H1:DCS-CALIB_STRAIN_C0X','L': 'L1:DCS-CALIB_STRAIN_C0X' ,'V': 'V1:Hrec_hoft_16384Hz'}.
        C02 is chossen as default given that is pressent in all observing runs.

    disposition: float (optional)
        Disposition is the time interval in seconds within we want the central times of the 
        signals to appear. It is used with differentSignals = True. The bigger
        the interval the bigger the spread. Although the spread is always constrained by
        the duration of the longest signa selected so that the signals don't get cropped.

    maxDuration : float (optional)
        If you want to use injections and want to restrict the maximum duration used.
        This will check the randomly selected injection's duration. If it is bigger
        that the one specified here, it will select another random injection. This will
        continue until it finds one with duration smaller than maxDuration parameter.
        Note that the smaller this parameter is the more time it will take to complete
        the generation.

    plugins : list of strings
        You can specify here plugins that are included in plugins.known_plug_ins variable.
        Those plugins will automaticly be added into the dataset returned.

    processingWindow : tuple/list
        Specify here where in the data window to search for gravitational wave signal. Data outside 
        this window is used for estimating the power spectrum.
        
    whitening_method : {'median','welch'} 
        The whitening method for any whitening, it accepts any gwpy methods.

    Returns
    -------

    A DataSet object. If savePath is defined, it saves the DataSet to that path
    and it returns None.



    """
    # Integration limits for the calculation of analytical SNR
    # These values are very important for the calculation
    fl, fm=20, int(fs/2)#

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
    # --- injection_source -------------------------------------------------------------------- #

    injection_source, inj_type = injection_initialization(injection_source, detectors)

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

    if injection_source is None :
        injectionSNR = 0

    input_injectionSNR=injectionSNR


    # ---------------------------------------------------------------------------------------- #   
    # --- noiseSourceFile -------------------------------------------------------------------- #

    if (backgroundType == 'sudo_real' or backgroundType =='real'):
        # if noiseSourceFile == None:
        #     raise TypeError('If you use sudo_real or real noise you need'
        #         +' a real noise file as a source.')

        # Loading noise using gwdatafind and gwpy
        if (isinstance(noiseSourceFile,list) 
                and len(noiseSourceFile)==len(detectors) 
                and all(len(el)==2 for el in noiseSourceFile)):

            noiseFormat='gwdatafind'

        # Loading noise using file paths of txt files (different for each detector)
        elif (isinstance(noiseSourceFile,list) 
              and len(noiseSourceFile)==len(detectors) 
              and all(isinstance(path_,str) for path_ in noiseSourceFile)
              and all(path_[-4:]=='.txt' for path_ in noiseSourceFile)):

            noiseFormat='txtD'

        # Loading noise using file paths of txt files (one with all detectors)
        elif (isinstance(noiseSourceFile,str) 
              and noiseSourceFile[-4:]=='.txt'):

            noiseFormat='txt1'

        # Loading noise using file paths of txt files (one with all detectors)
        elif (isinstance(noiseSourceFile,np.ndarray) and len(detectors) in noiseSourceFile.shape):

            noiseFormat='array'

        # Loding noise using DataPods or DataSets (one with all detectors)
        elif ((isinstance(noiseSourceFile,str) 
              and noiseSourceFile[-4:]=='.pkl') 
              or (('Pod' in str(type(noiseSourceFile))) or ('Set' in str(type(noiseSourceFile))))):

            noiseFormat='PodorSet'

        else:
            raise TypeError("The noise type format given is not one valid")



    # ---------------------------------------------------------------------------------------- #   
    # --- windowSize --(for PSD)-------------------------------------------------------------- #        

    if windowSize == None: windowSize = duration*16
    if not isinstance(windowSize,int):
        raise ValueError('windowSize needs to be a number')
    if windowSize < duration :
        raise ValueError('windowSize needs to be bigger than the duration')

    # ---------------------------------------------------------------------------------------- #   
    # --- timeSlides ------------------------------------------------------------------------- #

    # if timeSlides == None: timeSlides = 0
    # if not (isinstance(timeSlides, int) and timeSlides >=0) :
    #     raise ValueError('timeSlides has to be an integer equal or bigger than 0')

    # ---------------------------------------------------------------------------------------- #   
    # --- startingPoint ---------------------------------------------------------------------- #

    if startingPoint == None : startingPoint = 0 
    if not (isinstance(startingPoint, (float,int)) and (startingPoint%duration)%(1/fs) ==0) :
        raise ValueError('Starting point decimal part must always be a multiple of time step')        

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
            if pl in known_plug_ins:
                pass
            elif isinstance(pl,str) and ('knownPlugIns' in pl):
                pass
            else:
                raise TypeError("plugins must be a list of PlugIn object or from "+str(known_plug_ins))

    # --- fames ---

    if frames==None or (isinstance(frames,str) and frames.upper()=='C02'):
        frames = {'H': 'H1_HOFT_C02'
                 ,'L': 'L1_HOFT_C02'
                 ,'V': 'V1Online'}
    elif (isinstance(frames,str) and frames.upper()=='C01'):
        frames = {'H': 'H1_HOFT_C01'
                 ,'L': 'L1_HOFT_C01'
                 ,'V': 'V1Online'}
    elif (isinstance(frames,str) and frames.upper()=='C00'):
        frames = {'H': 'H1_HOFT_C00'
                 ,'L': 'L1_HOFT_C00'
                 ,'V': 'V1Online'}
    elif not (isinstance(frames,dict)): 

          raise ValueError("Frame type "+str(frames)+" is not valid")

    if channels==None or (isinstance(channels,str) and channels.upper()=='C02'):
        channels = {'H': 'H1:DCS-CALIB_STRAIN_C02'
                   ,'L': 'L1:DCS-CALIB_STRAIN_C02'
                   ,'V': 'V1:Hrec_hoft_16384Hz'}
    elif (isinstance(channels,str) and channels.upper()=='C01'):
        channels = {'H': 'H1:DCS-CALIB_STRAIN_C01'
                   ,'L': 'L1:DCS-CALIB_STRAIN_C01'
                   ,'V': 'V1:Hrec_hoft_16384Hz'}
    elif (isinstance(channels,str) and channels.upper()=='C00'):
        channels = {'H': 'H1:GDS-CALIB_STRAIN'
                   ,'L': 'L1:GDS-CALIB_STRAIN'
                   ,'V': 'V1:Hrec_hoft_16384Hz'}
    elif not (isinstance(channels,dict)): 

          raise ValueError("Channel type "+str(channels)+" is not valid")

    #print(frames,channels)

    
    # --- whitening_method -------------------------------------------- #

    if 'whitening_method' in kwargs:
        whitening_method = kwargs['whitening_method']
    else: 
        whitening_method = 'welch'
        
    # --- processingWindow -------------------------------------------- #
    
    if 'processingWindow' in kwargs:
        
        processingWindow = kwargs['processingWindow']
        
        if isinstance(processingWindow,(tuple,list)):

            if processingWindow[0]<0 or processingWindow[1]<0:
                raise ValueError('Expected positive values for the processing window.')

            if processingWindow[0]>windowSize or processingWindow[1]>windowSize:
                raise ValueError('Expected processing window to lie within data window.')

            if np.isclose(processingWindow[1]-processingWindow[0],duration, atol = 1-9):
                raise ValueError('Expected processing window to have same length as duration. Difference by '+str(processingWindow[1]-duration-processingWindow[0]))

        else: 
            raise TypeError('Processing window needs to be tuple or list within windowSize interval')
    else:
        processingWindow=((windowSize-duration)/2,(windowSize+duration)/2)

    #print('CHECKPOINT PW',processingWindow)
    

    # --- disposition


    # If you want no shiftings disposition will be zero
    if disposition == None: disposition=0
    # If you want shiftings disposition has to adjust the maximum length of an injection
    if isinstance(disposition,(int,float)) and 0<=disposition<duration:
        disposition_=disposition
        maxDuration_ = duration-disposition_
    # If you want random shiftings you can define a tuple or list of those random2*[[eventgps-windowSize/2,eventgps+windowSize/2]] shiftings.
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

    # --- max Duration

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

        if injection_source is None:
            injectionFileDict[det] = None
        elif inj_type == 'oldtxt':
            injectionFileDict[det] = dirlist(injection_source+'/' + det)

        elif inj_type == 'directory':
            injectionFileDict[det] = dirlist(injection_source)

        elif inj_type == 'DataPod':
            injectionFileDict[det] = [injection_source]

        elif inj_type == 'DataSet':
            injectionFileDict[det] = injection_source.dataPods

        else:
            raise TypeError("Unknown type of injections")



    # --- PSDm PSDc
    if 'PSDm' in kwargs: 
        PSDm=kwargs['PSDm']
    else:
        PSDm=None

    if 'PSDc' in kwargs: 
        PSDc=kwargs['PSDc']
    else:
        PSDc=None

    if PSDm==None: 
        PSDm={}
        for det in detectors:
            PSDm[det]=1
    if PSDc==None: 
        PSDc={}
        for det in detectors:
            PSDc[det]=0
    # ------------------------------
    # --- injectionHRSS ------------

    # I assume that whenever HRSS is used it is not going to be a noise case
    if 'injectionHRSS' in kwargs and injection_source is not None:
        injectionHRSS = kwargs['injectionHRSS']

        input_injectionHRSS=injectionHRSS

    else: 
        injectionHRSS=None
        input_injectionHRSS = None

    # ------------------------------
    if 'ignoreDetector' in kwargs: 
        ignoreDetector=kwargs['ignoreDetector']
    else:
        ignoreDetector=None


    # ------- profile ------

    if 'profile' in kwargs: 
        if isinstance(kwargs['profile'],list) and len(kwargs['profile'])>= len(detectors):
            profile = {}
            for d, detector in enumerate(detectors):
                profile[detector] = kwargs['profile'][d]
            
        else:
            
            raise ValueError("Profile needs to be defined for all the detectors.")
    else:

        profile = {'H' :'aligo','L':'aligo','V':'avirgo','K':'KAGRA_Early','I':'aligo'}
    
    print('profiles', profile)


    if backgroundType in ['sudo_real','real']:
        gps0 = {}
        if noiseFormat=='txtD':

            for det in detectors:
                noise_segDict[det] = np.loadtxt(noiseSourceFile[detectors.index(det)])

                gps0[det]=float(noiseSourceFile[detectors.index(det)].split('_')[1])

            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs
                                           -startingPoint-(windowSize-duration))
                               ,start_from_sec=startingPoint)

        elif noiseFormat=='txt1':

            file_=np.loadtxt(noiseSourceFile)

            if len(detectors) not in file_.shape: 
                raise ValueError(".txt file provided for noise doesn't have equal "
                                 +"to detector number entries of noise")

            for det in detectors:
                noise_segDict[det] = file_[detectors.index(det)]

                gps0[det]=float(noiseSourceFile.split('_')[1])

            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs
                                           -startingPoint-(windowSize-duration))
                               ,start_from_sec=startingPoint)

        elif noiseFormat=='array':

            file_=noiseSourceFile

            if len(detectors) not in file_.shape: 
                raise ValueError(".txt file provided for noise doesn't have equal "
                                 +"to detector number entries of noise")

            for det in detectors:
                noise_segDict[det] = file_[detectors.index(det)]

                gps0[det]=0

            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs
                                           -startingPoint-(windowSize-duration))
                               ,start_from_sec=startingPoint)

        elif noiseFormat=='PodorSet':

            if isinstance(noiseSourceFile,str):
                with open(noiseSourceFile,'rb') as obj:
                    file_ = pickle.load(obj)
            else:
                file_=noiseSourceFile

            if 'Pod' in str(type(file_)):
                noiseFormat = 'DataPod'
                for det in detectors:
                    noise_segDict[det] = file_.strain[detectors.index(det)]

                    gps0[det]=float(file_.gps[detectors.index(det)])


                ind=internalLags(detectors = detectors
                                   ,lags = timeSlides
                                   ,duration = duration
                                   ,fs = fs
                                   ,size = int(len(noise_segDict[detectors[0]])/fs
                                               -startingPoint-(windowSize-duration))
                                   ,start_from_sec=startingPoint)

                if size > int(len(noise_segDict[detectors[0]])/fs
                                               -startingPoint-(windowSize-duration)):
                    print("Requested size is bigger that the noise sourse data"
                                     +" can provide. Background will be used multiple times")

                    indexRepetition = ceil(size/int(len(noise_segDict[detectors[0]])/fs
                                               -startingPoint-(windowSize-duration)))

                    for det in detectors:
                        ind[det] = indexRepetition*ind[det]

            elif 'Set' in str(type(file_)):
                noiseFormat = 'DataSet'
                for podi in range(len(file_)):

                    if podi==0:
                        for det in detectors:
                            noise_segDict[det] = [file_[podi].strain[detectors.index(det)]]
                                
                            if len(file_[podi].strain[detectors.index(det)])!=windowSize*fs:
                                raise ValueError("Noise source data are not in the shape expected.")

                    else:
                        for det in detectors:
                            noise_segDict[det].append(file_[podi].strain[detectors.index(det)])

                for di, det in enumerate(detectors):
                    gps0[det] = np.asarray(file_.exportGPS())[:,di].astype('float')

                ind=internalLags(detectors = detectors
                                   ,lags = timeSlides
                                   ,duration = duration
                                   ,fs = 1
                                   ,size = len(file_)
                                   ,start_from_sec=startingPoint)

                if size > len(file_):
                    print("Requested size is bigger that the noise sourse data"
                                     +" can provide. Background will be used multiple times")

                    indexRepetition = ceil(size/len(file_))

                    for det in detectors:
                        ind[det] = indexRepetition*ind[det]
                        noise_segDict[det] = indexRepetition*noise_segDict[det]




        elif noiseFormat=='gwdatafind':
            for d in range(len(detectors)):

                # for trial in range(1):
                #     try:
                t0=time.time()
                conn=gwdatafind.connect()
                urls=conn.find_urls(detectors[d]
                                    , frames[detectors[d]]
                                    , noiseSourceFile[d][0]
                                    , noiseSourceFile[d][1])

                noise_segDict[detectors[d]]=TimeSeries.read(urls
                                                            , channels[detectors[d]]
                                                            , start =noiseSourceFile[d][0]
                                                            , end =noiseSourceFile[d][1]
                                                            ).resample(fs).astype('float64').value#[fs:-fs].value
                # Added [fs:fs] because there was and edge effect

                print("\n time to get "+detectors[d]+" data : "+str(time.time()-t0))

                if (len(np.where(noise_segDict[detectors[d]]==0.0)[0])
                    ==len(noise_segDict[detectors[d]])):
                    raise ValueError("Detector "+detectors[d]+" is full of zeros")
                elif len(np.where(noise_segDict[detectors[d]]==0.0)[0])!=0:
                    print("WARNING : "+str(
                        len(np.where(noise_segDict[detectors[d]]==0.0)[0]))
                            +" zeros were replased with the average of the array")
                print("Success on getting the "+str(detectors[d])+" data.")
                    #     break

                    # except Exception as e:
                    #     print(e.__class__,e)
                    #     print("/n")
                    #     print("Failed getting the "+str(detectors[d])+" data.\n")

                    #     #waiting=140+120*np.random.rand()
                    #     #os.system("sleep "+str(waiting))
                    #     #print("waiting "+str(waiting)+"s")
                    #     continue

                gps0[detectors[d]] = float(noiseSourceFile[d][0])

            ind=internalLags(detectors = detectors
                               ,lags = timeSlides
                               ,duration = duration
                               ,fs = fs
                               ,size = int(len(noise_segDict[detectors[0]])/fs-(windowSize-duration))
                               ,start_from_sec=startingPoint)

#             print(len(ind['H']),len(ind['L']))
#             print(len(noise_segDict['H'])/fs,len(noise_segDict['L'])/fs)



    thetime = time.time()
    DATA=DataSet(name = name)

    for I in range(size):


        t0=time.time()
        
        if isinstance(input_injectionSNR,(list,tuple)):
            injectionSNR = np.random.uniform(input_injectionSNR[0],input_injectionSNR[1])

        elif callable(input_injectionSNR):
            injectionSNR = input_injectionSNR()

        elif input_injectionSNR=='same':
            injectionSNR = 'same'
            print('INJECTION SNR SET TO SAME')



        
        if isinstance(input_injectionHRSS,(list,tuple)):
            injectionHRSS = np.random.uniform(input_injectionHRSS[0],input_injectionHRSS[1])

        elif callable(input_injectionHRSS):
            injectionHRSS = input_injectionHRSS()


        detKeys = list(injectionFileDict.keys())

        if single == True: luckyDet = np.random.choice(detKeys)
        if isinstance(disposition,(list,tuple)):
            disposition_=disposition[0]+np.random.rand()*(disposition[1]-disposition[0])
            maxDuration_=min(duration-disposition_,maxDuration)

        index_selection={}
        if injection_source != None:

            if differentSignals==True:
                if maxDuration_ != duration:
                    for det in detectors: 
                        index_sample=np.random.randint(0,
                                                  len(injectionFileDict[detKeys[0]]))

                        if inj_type =='oldtxt':
                            sampling_strain=np.loadtxt(injection_source+'/'
                                                       +det+'/'+injectionFileDict[det][index_sample])

                        elif inj_type == 'directory':
                            s_pod=DataPod.load(injection_source+
                                                         '/'+injectionFileDict[det][index_sample])
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                        # case where we pass one pod only
                        elif inj_type == 'DataPod':
                            raise TypeError("You cannot have single pod for inejctions and "
                                            +" also differentSignals = True")

                        elif inj_type == 'DataSet':
                            s_pod = injectionFileDict[det][index_sample] 
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                        index_selection[det]=index_sample

                        while (len(sampling_strain)/fs > maxDuration_ 
                               or list(index_selection.values()).count(index_sample) > 1 ):

                            index_sample=np.random.randint(0,
                                                      len(injectionFileDict[detKeys[0]]))

                            if inj_type=='oldtxt':
                                sampling_strain=np.loadtxt(injection_source+'/'
                                                       +det+'/'+injectionFileDict[det][index_sample])

                            elif inj_type == 'directory':
                                s_pod=DataPod.load(injection_source+
                                                         '/'+injectionFileDict[det][index_sample])
                                sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                            elif inj_type == 'DataSet':
                                s_pod = injectionFileDict[det][index_sample] 
                                sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                            index_selection[det]=index_sample

                else:
                    for det in detectors: 
                        index_sample=np.random.randint(0,
                                                  len(injectionFileDict[detKeys[0]]))

                        index_selection[det]=index_sample

                        while (list(index_selection.values()).count(index_sample) > 1):

                            index_sample=np.random.randint(0,
                                                      len(injectionFileDict[detKeys[0]]))

                            index_selection[det]=index_sample
            else:
                if maxDuration_ != duration:
                    index_sample=np.random.randint(0,len(injectionFileDict[detKeys[0]]))

                    if inj_type =='oldtxt':
                        sampling_strain=np.loadtxt(injection_source+'/'
                                                   +det+'/'+injectionFileDict[det][index_sample])

                    elif inj_type == 'directory':
                        s_pod=DataPod.load(injection_source+
                                                     '/'+injectionFileDict[det][index_sample])
                        sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                    elif inj_type == 'DataPod':
                        s_pod=injection_source
                        sampling_strain=s_pod.strain[s_pod.detectors.index(det)]         

                    elif inj_type == 'DataSet':
                        s_pod = injectionFileDict[det][index_sample] 
                        sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                    while (len(sampling_strain)/fs > maxDuration_
                           or index_sample in list(index_selection.values())):

                        index_sample=np.random.randint(0,len(injectionFileDict[detKeys[0]]))

                        if inj_type =='oldtxt':
                            sampling_strain=np.loadtxt(injection_source+'/'
                                                       +det+'/'+injectionFileDict[det][index_sample])

                        elif inj_type == 'directory':
                            s_pod=DataPod.load(injection_source+
                                                         '/'+injectionFileDict[det][index_sample])
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

                        elif inj_type == 'DataPod':
                            s_pod=injection_source
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]         

                        elif inj_type == 'DataSet':
                            s_pod = injectionFileDict[det][index_sample] 
                            sampling_strain=s_pod.strain[s_pod.detectors.index(det)]

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
                PSD,X,T=simulateddetectornoise(profile[det]
                                              ,windowSize
                                              ,fs,10,fs/2
                                              ,PSDm=PSDm[det]
                                              ,PSDc=PSDc[det])
                # Calculatint the psd of FFT=1s
                p, f = psd(X, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSD=psd_int(np.arange(0,fs/2,1/windowSize))
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
                noise=noise_seg[int(ind[det][I]):int(ind[det][I])+windowSize*fs]  
                # Generating the PSD of it
                p, f = psd(noise, Fs=fs, NFFT=fs) 
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSD,X,T=simulateddetectornoise([f,p]
                                               ,windowSize,fs,10
                                               ,fs/2,PSDm=PSDm[det]
                                               ,PSDc=PSDc[det])
                p, f = psd(X, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSD=psd_int(np.arange(0,fs/2,1/windowSize))
                PSD_dict[det]=PSD
                back_dict[det] = X
                # Making the noise a TimeSeries
                back=TimeSeries(X,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asd=back.asd(1,0.5)                 
                asd_dict[det] = asd
                gps_list.append(gps0[det]+ind[det][I]/fs+processingWindow[0])
                back_dict[det] = back.value

            elif backgroundType == 'real':
                noise_seg=noise_segDict[det]
                # Calling the real noise segments

                if noiseFormat == 'DataSet':
                    noise=noise_seg[I]

                else:
                    noise=noise_seg[int(ind[det][I]):int(ind[det][I])+windowSize*fs]
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
                #print(det,back,len(back),type(back))
                asd=back.asd(1,0.5)
                asd_dict[det] = asd
                if noiseFormat == 'DataSet':
                    gps_list.append(gps0[det][I]+processingWindow[0])
                else:
                    gps_list.append(gps0[det]+ind[det][I]/fs+processingWindow[0])

            #If this dataset includes injections:            
            if injection_source != None:      

                # Calling the templates generated with PyCBC
                # OLD inj=load_inj(injection_source,injectionFileDict[det][inj_ind], det) 
                if inj_type =='oldtxt':
                    inj = np.loadtxt(injection_source+'/'
                                     +det+'/'+injectionFileDict[det][index_selection[det]])

                elif inj_type == 'directory':
                    inj_pod=DataPod.load(injection_source+'/'+injectionFileDict[det][index_selection[det]])
                    inj=np.array(inj_pod.strain[inj_pod.detectors.index(det)])


                elif inj_type == 'DataPod':
                    inj_pod=injection_source
                    inj=np.array(inj_pod.strain[inj_pod.detectors.index(det)])

                elif inj_type == 'DataSet':
                    inj_pod = injectionFileDict[det][index_selection[det]]
                    inj=np.array(inj_pod.strain[inj_pod.detectors.index(det)])

                if inj_type in ['DataSet','DataPod','directory']:

                    if injectionHRSS!=None:
                        if 'hrss' in inj_pod.pluginDict.keys():
                            hrss0=inj_pod.hrss
                        else:
                            raise AttributeError("There is no hrss in the injection pod")
                    else:
                        if 'hrss' in inj_pod.pluginDict.keys():
                            hrss0=inj_pod.hrss


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
                    inj = np.hstack((np.zeros(int(fs*processingWindow[0])),inj[disp:]
                                         ,np.zeros(int(fs*(windowSize-processingWindow[1]))+disp))) 
                if disp < 0: 
                    inj = np.hstack((np.zeros(int(fs*processingWindow[0])-disp),inj[:disp]
                                         ,np.zeros(int(fs*(windowSize-processingWindow[1]))))) 


                # Calculating the one sided fft of the template,
                # Norm default is 'backwards' which means that it normalises with 1/N during IFFT and not duriong FFT

                inj_fft_0=(1/fs)*np.fft.fft(inj)
                inj_fft_0_dict[det] = inj_fft_0

                # Getting rid of negative frequencies and DC
                inj_fft_0N= inj_fft_0[1:int(windowSize*fs/2)+1]


                SNR0_dict[det]=np.sqrt((1/windowSize #(fs/N=fs/fs*windowSize 
                                       )*4*np.sum( # 2 from PSD, S=1/2 S1(one sided) and 2 from integral symetry
                    np.abs(inj_fft_0N*inj_fft_0N.conjugate())[windowSize*fl-1:windowSize*fm-1]
                                               /PSD_dict[det][windowSize*fl-1:windowSize*fm-1]))

                if single == True:
                    if det != luckyDet:
                        SNR0_dict[det] = 0.01 # Making single detector injections as glitch
                        inj_fft_0_dict[det] = np.zeros(windowSize*fs)


            else:
                SNR0_dict[det] = 0.01 # avoiding future division with zero
                inj_fft_0_dict[det] = np.zeros(windowSize*fs)


#             print(
#                 list(
#                     len(np.where(
#                         noise_segDict[det][int(ind[det][I]):int(ind[det][I])+windowSize*fs]
#                     )[0]
#                        )
#                     for det in detKeys
#                 )
#             )
        if (backgroundType=='real' 
            and any(len(np.where(noise_segDict[det][
                int(ind[det][I]):int(ind[det][I])+windowSize*fs]==0)[0])==windowSize*fs for det in detKeys)):
            print(I,"skipped")
            continue

        # Calculation of combined SNR    
        SNR0=np.sqrt(np.sum(np.asarray(list(SNR0_dict.values()))**2))
        # Tuning injection amplitude to the SNR wanted
        podstrain = []
        unwhitened_strain = []
        podPSD = []
        podCorrelations=[]
        SNR_new=[]
        SNR_new_sq_sum=0
        psdPlugDict={}
        plugInToApply=[]



        for det in detectors:
            if injectionHRSS!=None:
                fft_cal=(injectionHRSS/hrss0)*inj_fft_0_dict[det]     
    
            else:
                if injectionSNR=='same':
                    fft_cal=inj_fft_0_dict[det] 
                else:
                    fft_cal=(injectionSNR/SNR0)*inj_fft_0_dict[det] 

            # Norm default is 'backwards' which means that it normalises with 1/N during IFFT and not duriong FFT
            if ignoreDetector ==None:
                inj_cal=np.real(np.fft.ifft(fs*fft_cal)) 
            elif ignoreDetector==det:
                inj_cal=0.0001*np.real(np.fft.ifft(fs*fft_cal)) 
            else:
                inj_cal=np.real(np.fft.ifft(fs*fft_cal)) 
                
            # Joining calibrated injection and background noise
            strain= TimeSeries(back_dict[det]+inj_cal,sample_rate=fs,t0=0).astype('float64')
            unwhitened_strain.append(strain.value.tolist())
            #print(det,len(strain),np.prod(np.isfinite(strain)),len(strain)-np.sum(np.isfinite(strain)))
            #print(det,len(strain),'zeros',len(np.where(strain.value==0.0)[0]))
            #print(strain.value.tolist())
            # Bandpassing
            # strain=strain.bandpass(20,int(fs/2)-1)
            # Whitenning the data with the asd of the noise
            whiten_strain=strain.whiten(4,2,fduration=4,method = whitening_method, highpass=20)#,asd=asd_dict[det])

            #print(det,len(strain),np.prod(np.isfinite(strain)),len(strain)-np.sum(np.isfinite(strain)))
            #print(det,len(strain),'zeros',len(np.where(strain.value==0.0)[0]))

            # Crop data to the duration length
            whiten_strain=whiten_strain[int((processingWindow[0])*fs):int((processingWindow[1])*fs)]
            podstrain.append(whiten_strain.value.tolist())

            if 'snr' in plugins:
                whiten_strain_median=strain.whiten(4,2,fduration=4,method = whitening_method
                        , highpass=20)[int((processingWindow[0])*fs):int((processingWindow[1])*fs)]
                # This is strictly for duration 1 and fs 1024
                new_snr = np.sum(whiten_strain_median.value**2 -0.978**2)

                SNR_new.append(np.sqrt(max(new_snr,0)))
                SNR_new_sq_sum += new_snr


            # if 'snr' in plugins:
            #     # Adding snr value as plugin.
            #     # Calculating the new SNR which will be slightly different that the desired one.    
            #     inj_fft_N=fft_cal[1:int(windowSize*fs/2)+1]
            #     SNR_new.append(np.sqrt((1/windowSize #(fs/fs*windowsize
            #                            )*4*np.sum( # 2 from integral + 2 from S=1/2 S1(one sided)
            #         np.abs(inj_fft_N*inj_fft_N.conjugate())[windowSize*fl-1:windowSize*fm-1]
            #                                  /PSD_dict[det][windowSize*fl-1:windowSize*fm-1])))

            #     plugInToApply.append(PlugIn('snr'+det,SNR_new[-1]))

            if 'psd' in plugins:
                podPSD.append(asd_dict[det]**2)

        if 'snr' in plugins: 

            network_snr = np.sqrt(max(SNR_new_sq_sum,0))
            SNR_new.append(network_snr)
            plugInToApply.append(PlugIn('snr',SNR_new))

        if 'uwstrain' in plugins:
            uw_strain = PlugIn('uwstrain', np.array(unwhitened_strain))
            plugInToApply.append(uw_strain)



        if 'psd' in plugins:
            plugInToApply.append(PlugIn('psd'
                                        ,genFunction=podPSD
                                        ,plotFunction=plotpsd
                                        ,plotAttributes=['detectors','fs']))

        for pl in plugins:
            if 'correlation' in pl:
                plugInToApply.append(knownPlugIns(pl))



        pod = DataPod(strain = podstrain
                           ,fs = fs
                           ,labels =  labels
                           ,detectors = detKeys
                           ,gps = gps_list
                           ,duration = duration)

        if injection_source!=None and inj_type in ['DataSet','DataPod','directory']:

            for plkey in list(inj_pod.pluginDict.keys()):
                if not (plkey in list(pod.pluginDict.keys())):
                    pod.addPlugIn(inj_pod.pluginDict[plkey])

        if 'hrss' in plugins or injectionHRSS!=None:
            if 'hrss' in inj_pod.pluginDict.keys():
                if injectionHRSS!=None:
                    plugInToApply.append(PlugIn('hrss'
                                            ,genFunction=inj_pod.hrss*(injectionHRSS/hrss0)))
                else:
                    if injectionSNR == 'same':
                        plugInToApply.append(PlugIn('hrss'
                                                ,genFunction=inj_pod.hrss))
                    else:                        
                        plugInToApply.append(PlugIn('hrss'
                                                ,genFunction=inj_pod.hrss*(injectionSNR/SNR0)))

            else:
                raise ValueError("Unable to calculate or use hrss valye. There was no hrss in the injection pod.")

        for pl in plugInToApply:
            pod.addPlugIn(pl)

        DATA.add(pod)

        #t1=time.time()
        #sys.stdout.write("\r Instantiation %i / %i --- %s" % (I+1, size, str(t1-t0)))
        #sys.stdout.flush()
        #t0=time.time()

    if not ('shuffle' in kwargs):
        kwargs['shuffle'] = True

    if not isinstance(kwargs['shuffle'],bool):
        raise TypeError("shuffle option must be a bool value")

    if kwargs['shuffle']==True:
         random.shuffle(DATA.dataPods)
    # else if false do not shuffle.

    print('\n')
    if savePath!=None:
        DATA.save(savePath+'/'+name,'pkl')
    else:
        return(DATA)



    

def auto_gen(duration 
             ,fs
             ,detectors
             ,size
             ,injection_source = None
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
             ,plugins=None
             ,finalDirectory=None
             ,**kwargs): 




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
    # --- injection_source -------------------------------------------------------------------- #

    if injection_source == None:
        pass
    elif isinstance(injection_source, str):            
        if (('/' in injection_source) and os.path.isdir(injection_source)):
            injection_source_set = injection_source.split('/')[-1]
        elif (('/' not in injection_source) and any('injections' in p for p in sys.path)):
            for p in sys.path:
                if ('injections' in p):
                    injection_source_path = (p.split('injections')[0]+'injections'
                        +'/'+injection_source)
                    if os.path.isdir(injection_source_path):
                        injection_source = "'"+injection_source_path+"'"
                        injection_source_set = injection_source.split('/')[-1]
                    else:
                        raise FileNotFoundError('No such file or directory:'
                                                +injection_source_path)

        else:
            raise FileNotFoundError('No such file or directory:'+injection_source) 
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
    if (any(snr != 0 for snr in injectionSNR)  and  injection_source == None):
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

    if windowSize == None: windowSize = duration*16
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
    
    # Accounting group options
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
        
    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o4.burst.allsky.mlyonline'
        print("Accounting group set to 'ligo.dev.o4.burst.allsky.mlyonline")
    
   
    # The number of sets to be generated.
    num_of_sets = len(injectionSNR)

    # If noise is optimal it is much more simple
    if backgroundType == 'optimal':

        if isinstance(snr_list[0],(int,float,str)):

            d={'size' : num_of_sets*[size]
            , 'start_point' : num_of_sets*[startingPoint]
            , 'set' : snr_list
            , 'name' : list(name+'_'+str(snr_list[i]) for i in range(num_of_sets))}

            print('These are the details of the datasets to be generated: \n')
            for i in range(len(d['size'])):
                print(d['size'][i], d['start_point'][i] ,d['name'][i])

        elif isinstance(snr_list[0],(list,tuple)):

            d={'size' : num_of_sets*[size]
            , 'start_point' : num_of_sets*[startingPoint]
            , 'set' : snr_list
            , 'name' : list(name+'_'+str(snr_list[i][0])+'to'+str(snr_list[i][1]) for i in range(num_of_sets))}

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
    if finalDirectory==None:
        answer=None
    else:
        answer='y'
        
    while answer not in answers:
        print('Should we proceed to the generation of the following'
              +' data y/n ? \n \n')
        answer=input()
        if answer not in answers: print("Not valid answer ...")
    
    if answer in ['no','n', 'No','NO','N','exit']:
        print('Exiting procedure ...')
        return
    elif answer in ['yes','y','YES','Yes','Y']:
        if finalDirectory==None:
            print('Type the name of the temporary directory:')
            dir_name = '0 0'
        else:
            dir_name = finalDirectory
        
        while not dir_name.isidentifier():
            dir_name=input()
            if not dir_name.isidentifier(): print("Not valid Folder name ...")
        
    path = savePath+'/'
    print("The current path of the directory is: \n"+path+dir_name+"\n" )  
    if finalDirectory==None:
        answer=None
    else:
        answer='y'
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
    
    kwstr=""
    for k in kwargs:
        if isinstance(kwargs[k],str):
            kwstr+=(","+k+"='"+str(kwargs[k])+"'")
        else:
            kwstr+=(","+k+"="+str(kwargs[k])) 

    for i in range(len(d['size'])):

        with open(path+dir_name+'/'+'gen_'+d['name'][i]+'_'
            +str(d['size'][i])+'.py','w') as f:
            f.write('#!'+which_python+'\n')
            f.write('import sys \n')

            f.write('from mly.datatools import DataPod, DataSet\n\n')

            if isinstance(d['set'][i],(float,int)):
                token_snr = str(d['set'][i])
            else:
                token_snr = '0'
            f.write("import time\n\n")
            f.write("t0=time.time()\n")
            
            if injection_source!=None and injection_source[0]!="'":
                injection_source = "'"+injection_source+"'"

            if backgroundType == 'optimal':
                comand=( "SET = generator(\n"
                         +24*" "+"duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         +24*" "+",size = "+str(d['size'][i])+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",injection_source ="+str(injection_source)+"\n"
                         +24*" "+",labels = "+str(labels)+"\n"
                         +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                         +24*" "+",injectionSNR = "+token_snr+"\n"
                         +24*" "+",windowSize = "+str(windowSize)+"\n"
                         +24*" "+",name = '"+str(d['name'][i])+"_"+str(d['size'][i])+"'\n"
                         +24*" "+",savePath ='"+path+dir_name+"'\n"
                         +24*" "+",single = "+str(single)+"\n"
                         +24*" "+",injectionCrop = "+str(injectionCrop)+"\n"
                         +24*" "+",differentSignals = "+str(differentSignals)+"\n"
                         +24*" "+",plugins = "+str(plugins)+kwstr+")\n")

            else:
                f.write("sys.path.append('"+date_list_path[:-1]+"')\n")
                comand=( "SET = generator(\n"
                         +24*" "+"duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         +24*" "+",size = "+str(d['size'][i])+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",injection_source = "+str(injection_source)+"\n"
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
                         +24*" "+",plugins = "+str(plugins)+kwstr+")\n")

            
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
               ,retry=10
               ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"] )

        job_list.append(job)

    with open(path+dir_name+'/info.txt','w') as f3:
        f3.write('INFO ABOUT DATASETS GENERATION \n\n')
        f3.write('fs: '+str(fs)+'\n')
        f3.write('duration: '+str(duration)+'\n')
        f3.write('window: '+str(windowSize)+'\n')
        if injection_source!=None:
            f3.write('injection_source: '+str(injection_source)+'\n')
        
        if backgroundType != 'optimal':
            f3.write('timeSlides: '+str(timeSlides)+'\n'+'\n')
            for i in range(len(d['size'])):
                f3.write(d['segment'][i][0]+' '+d['segment'][i][1]
                         +' '+str(d['size'][i])+' '
                         +str(d['start_point'][i])+'_'+d['name'][i]+'\n')
    with open(path+dir_name+'/final_gen.py','w') as f4:
        f4.write('#!'+which_python+'\n')
        f4.write("import sys \n")
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
               ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])
    
    final_job.add_parents(job_list)
    
    if finalDirectory==None:
        print('All set. Initiate dataset generation y/n?')
        answer4=input()
    else:
        answer4='y'


    if answer4 in ['yes','y','YES','Yes','Y']:
        print('Creating Job queue')
        
        dagman.build_submit()

        return
    
    else:
        print('Data generation canceled')
        os.system('cd')
        os.system('rm -r '+path+dir_name)
        return


    
    
def finalise_gen(path,generation=True,**kwargs):
    
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
                
    # Accounting group options
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
        
    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o4.burst.allsky.mlyonline'
        print("Accounting group set to 'ligo.dev.o4.burst.allsky.mlyonline")
    
    
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
                           ,retry=10
                           ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])

                repeat_job_list.append(repeat_job)

               
        repeat_final_job = Job(name='repeat_finishing'
                           ,executable=path+'finalise_gen.py'
                           ,submit=submit
                           ,error=error
                           ,output=output
                           ,log=log
                           ,getenv=True
                           ,dag=repeat_dagman
                           ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])

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




def stackDetector(dataset,**kwargs):
    kwargs['size']=len(dataset)
    if 'duration' not in kwargs: kwargs['duration']=dataset[0].duration
    if 'fs' not in kwargs: kwargs['fs']=dataset[0].fs   
    if 'detectors' not in kwargs: 
        raise ValueError("You need to at least specify a detector")

    if 'plugins' not in kwargs: kwargs['plugins']=[]
    if 'psd' in dataset[0].pluginDict and 'psd' not in kwargs['plugins']: kwargs['plugins'].append('psd')
    # if 'snr'+dataset[0].detectors[0] in dataset[0].pluginDict and 'snr' not in kwargs['plugins']: 
    #     kwargs['plugins'].append('snr')
    

    newSet=generator(**kwargs)


    for i in range(len(dataset)):
        dataset[i].strain=np.vstack((dataset[i].strain,newSet[i].strain))

        if isinstance(dataset[i].detectors,str):
            dataset[i].detectors = list(dataset[i].detectors) + newSet[i].detectors
        else:
            dataset[i].detectors = dataset[i].detectors + newSet[i].detectors


        dataset[i].gps+=newSet[i].gps

        if 'psd' in kwargs['plugins']: dataset[i].psd+=newSet[i].psd
        # if 'snr' in kwargs['plugins']:
        #     for d in newSet[i].detectors:
        #         dataset[i].addPlugIn(newSet[i].pluginDict['snr'+d])        
        if 'correlation' in dataset[i].pluginDict:
            dataset[i].addPlugIn(dataset[i].pluginDict['correlation'])   
            
    return(dataset)
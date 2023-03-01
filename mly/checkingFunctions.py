import os
from .tools import dirlist
from .datatools import DataPod, DataSet

# duration
def check_duration(duration):
    if not (isinstance(duration,(float,int)) and duration>0 ):
        raise ValueError('The duration value has to be a possitive float'
                        +' or integer representing seconds.')

# fs

def check_fs(fs):
    if not (isinstance(fs,int) and fs>0):
        raise ValueError('Sample frequency has to be a positive integer.')
        

# detectors

def check_detectors(detectors):
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
    return detectors
        
def check_size(size, stringReturn = True):
    if not (isinstance(size, int) and size > 0):
        raise ValueError("size must be a possitive integer.")
        
def check_windowSize(windowSize, duration):
    if windowSize == None: windowSize = duration*16
    
    if not isinstance(windowSize,int):
        raise ValueError('windowSize needs to be an integer number')
    if windowSize < duration :
        raise ValueError('windowSize needs to be bigger than the duration')
    
    return windowSize

def check_dates(dates):
    
    from gwpy.time import to_gps

    gps_start = to_gps(dates[0])
    gps_end = to_gps(dates[1])
    
    if gps_start > gps_end:
        raise ValueError("Not valid date.")
    else:
        return (gps_start, gps_end)
    
def check_observingFlags(observingFlags=None, **kwargs):
            
    if (observingFlags == None and 'observingFlags' in kwargs):
        obsrevingFlags = kwargs['observingFlags']
    elif (observingFlags == None and 'observingFlags' not in kwargs):
        observingFlags = [{'H1': 'H1:DMT-ANALYSIS_READY:1'
                          ,'L1': 'L1:DMT-ANALYSIS_READY:1'
                          ,'V1': 'V1:ITF_SCIENCE:1'}]
        
    if isinstance(observingFlags,dict):
        observingFlags = [observingFlags]
 
    elif not (isinstance(observingFlags,list) 
        and all(isinstance(flag,dict) for flag in observingFlags)):
        
        raise TypeError("observingFlags must be a list of dictionaries "
                        "or a dictionary")
        
    return observingFlags


def check_excludedFlags(excludedFlags=None, **kwargs):
    
    if (excludedFlags == None and 'excludedFlags' in kwargs):
        obsrevingFlags = kwargs['observingFlags']
    elif (excludedFlags == None and 'excludedFlags' not in kwargs):


        cbc_inj_flags={'H1': 'H1:ODC-INJECTION_CBC:2'
                      ,'L1': 'L1:ODC-INJECTION_CBC:2'}

        burst_inj_flags={'H1': 'H1:ODC-INJECTION_BURST:2'
                        ,'L1': 'L1:ODC-INJECTION_BURST:2'}

        detchar_inj_flags={'H1': 'H1:ODC-INJECTION_DETCHAR:2'
                          ,'L1': 'L1:ODC-INJECTION_DETCHAR:2'}

        stoch_inj_flags={'H1': 'H1:ODC-INJECTION_STOCHASTIC:2'
                        ,'L1': 'L1:ODC-INJECTION_STOCHASTIC:2'}

        trans_inj_flags={'H1': 'H1:ODC-INJECTION_TRANSIENT:2'
                        ,'L1': 'L1:ODC-INJECTION_TRANSIENT:2'}

        excludedFlags=[cbc_inj_flags
                       ,burst_inj_flags
                       ,detchar_inj_flags
                       ,stoch_inj_flags
                       ,trans_inj_flags]

    if (isinstance(excludedFlags,dict)):
        excludedFlags = [excludedFlags]

    elif not (isinstance(excludedFlags,list) 
          and all(isinstance(flag,dict) for flag in excludedFlags)):

        raise TypeError("excludedFlags must be a list of dictionaries ")
        
    return(excludedFlags)

def check_maxSegmentSize(maxSegmentSize=None,duration=None,windowSize=None,**kwargs):

    if duration==None or windowSize==None:
        raise ValueError("duration and windowSize must be defined for this check.")
        
    if maxSegmentSize == None and 'maxSegmentSize' in kwargs:
        maxSegmentSize = kwarg['maxSegmentSize']
    elif maxSegmentSize == None and 'maxSegmentSize' not in kwargs:
        maxSegmentSize = int(3600/duration)
        

    if not (isinstance(maxSegmentSize,int) and maxSegmentSize >= windowSize):
        raise ValueError("maxSegmentSize must be an integer at least equal to"
                         " windowSize")
        
    return(maxSegmentSize)


def check_labels(labels):
    if labels == None:
        labels = {'type' : 'UNDEFINED'}
    elif not isinstance(labels,dict):
        raise TypeError(" Labels must be a dictionary.Suggested keys for labels"
                        +"are the following: \n{ 'type' : ('noise' , 'cbc' , 'signal'"
                        +" , 'burst' , 'glitch', ...),\n'snr'  : ( any int or float number "
                        +"bigger than zero),\n'delta': ( Declination for sky localisation,"
                        +" float [-pi/2,pi/2])\n'ra'   : ( Right ascention for sky "
                        +"localisation float [0,2pi) )})")
    return labels

def check_masterDirectory_createFS(masterDirectory, detectors):

    if masterDirectory==None:
        raise TypeError("You need to define a masterDirectory path")
    elif isinstance(masterDirectory, str):
        
        if masterDirectory[-1]!="/": masterDirectory+="/"
        
        if os.path.isdir(masterDirectory):
            print("masterDirectory already exists, files might be replased")
        
        else:
            os.system("mkdir "+masterDirectory)
            for det in detectors:
                os.system("mkdir "+masterDirectory+"/"+det)
                
    else:
        raise TypeError("masterDirectory must be a path in string format")
        
    return masterDirectory


def check_masterDirectory_verifyFS(masterDirectory, detectors):
    
    if masterDirectory==None:
        raise TypeError("You need to define a masterDirectory path")
    elif isinstance(masterDirectory, str):
        
        if masterDirectory[-1]!="/": masterDirectory+="/"
        
        if not (os.path.isdir(masterDirectory)
                and all(os.path.isdir(masterDirectory+det) for det in detectors)):
            raise FileNotFoundError("Filesystem structure not correct")
            
        fileSizes=list(len(dirlist(masterDirectory+det)) for det in detectors)
        result = 1
        for x in fileSizes:
            result = result * x

        if result!= len(dirlist(masterDirectory+detectors[0]))**(len(detectors)):
            print("WARNING: Directories don't have the same amount of files, "
                  "with number of files "+str(fileSizes)+" each.")
            
    else:
        raise TypeError("masterDirectory must be a path in string format")
        
    return masterDirectory

def check_backgroundType(backgroundType, **kwargs):
    if backgroundType == None and 'backgroundType' in kwargs:
        backgroundType = kwargs['backgroundType']
    elif backgroundType == None and 'backgroundType' not in kwargs:
        backgroundType = 'optimal'
        
    if not (isinstance(backgroundType,str) 
          and (backgroundType in ['optimal','sudo_real','real'])):
        raise ValueError("backgroundType is a string that can take values : "
                        +"'optimal' | 'sudo_real' | 'real'.")
        
    return backgroundType

def check_frames(frames=None, detectors=None, **kwargs):
    
    if detectors==None:
        raise ValueError("detectors must be defined for this check.")
       
    if (frames==None and 'frames' in kwargs):
        frames = kwargs['frames']
    elif (frames==None and 'frames' not in kwargs):
        frames='C02'
        
    if (isinstance(frames,str) and frames.upper()=='C02'):
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

    return frames

def check_channels(channels=None, detectors=None, **kwargs):
    
    if detectors==None:
        raise ValueError("detectors must be defined for this check.")
    
    if (channels==None and 'channels' in kwargs):
        channels = kwargs['channels']
    elif (channels==None and 'channels' not in kwargs):
        channels='C02'
        
    if (isinstance(channels,str) and channels.upper()=='C02'):
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
    
          raise ValueError("Frame type "+str(channels)+" is not valid")

    return channels


def check_kw_accounting_group(**kwargs):

    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o3.burst.grb.xoffline'
        print("Accounting group set to 'ligo.dev.o3.burst.grb.xoffline")
        
    return accounting_group

def check_kw_accounting_group_user(**kwargs):
    
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
    
    return accounting_group_user


def check_models(models, returnTrainedModels=False, **kwargs):
    
    from tensorflow.keras.models import load_model
    
    if not isinstance(models,list):
        models=[[models],[None]]
    # Case where all models have all data to use
    if isinstance(models,list) and not all(isinstance(m,list) for m in models):
        models=[models,len(models)*[None]]
    # Case where index is not given for all models.
    if len(models[0])!=len(models[1]):
        raise ValueError('You have to define input index for all models')
    # Case somebody doesn't put the right amount of indexes for the data inputs. 

    if not (isinstance(models,list) and all(isinstance(m,list) for m in models)):
        raise TypeError('models have to be a list of two sublists. '
                        +'First list has the models and the second has the'
                        +' indexes of the data each one uses following the order strain, extra1, extra2...'
                        +'[model1,model2,model3],[[0,1],[0,2],[2]]')
        
    if returnTrainedModels==False:
        return models
    else:
        # models[0] becomes the trainedModels list
        trained_models=[]
        for model in models[0]:  
            if isinstance(model,str):
                if os.path.isfile(model):    
                    trained_models.append(load_model(model))
                else:
                    raise FileNotFoundError("No model file in "+model)
            else:
                trained_models.append(model) 
                
        return(models, trained_models)
    


def check_dataSets_asinput(dataSets, detectors=None, windowSize=None, masterDirectory=None,**kwargs):
    
    if detectors==None or masterDirectory==None:
        raise ValueError("detectors and masterDirectory must be defined for this check.")
    if 'verbose' in kwargs and kwargs['verbose']==True :
        verbose=kwargs['verbose']
    else:
        verbose=False

        
    filelist=dirlist(masterDirectory+"/"+detectors[0])
    firstfile=DataSet.load(masterDirectory+"/"+detectors[0]+"/"+filelist[0])
    

    if len(firstfile)==0: raise ValueError("DataSet provided is empty")

    duration=firstfile[0].duration
    fs=firstfile[0].fs


    dataset_dict={}

    for det in detectors:

        if not isinstance(dataSets,list):
            dataSets=[dataSets]

        if isinstance(dataSets,list):

            # Case where we provide a gps interval
            if len(dataSets)==2 and all(isinstance(el,(int,float)) for el in dataSets):
                filesToUse=[]
                indecesToUse=[]
                # requested gps range
                startGPS, endGPS = dataSets[0], dataSets[1]
                
                for file in filelist:
                    # the start gps of the first instance of the file 
                    fileStartGPS=int(file.split('-')[0])+(windowSize-duration)/2
                    # the start gps of the last instance of the file 
                    fileEndGPS=int(file.split('-')[1].split('_')[0])-(windowSize+duration)/2
                    fileDuration=int(file.split('_')[1].split('.')[0])

                    # "[]" requested gps limits
                    # "()" file gps limits

                    # ---[---(----]---)---
                    if startGPS<fileStartGPS and fileStartGPS<endGPS<fileEndGPS:
                        filesToUse.append(file)
                        indecesToUse.append([0, int(fileDuration-(fileEndGPS-endGPS))])
                        if verbose : print('case1')
                        break

                    # ---(---[----]---)---
                    elif fileStartGPS<=startGPS<fileEndGPS and fileStartGPS<endGPS<=fileEndGPS:
                        filesToUse.append(file)
                        indecesToUse.append([int(startGPS-fileStartGPS)
                                             , int(fileDuration-(fileEndGPS-endGPS))])
                        if verbose : print('case2')
                        break
                    # ---[---(----)---]---
                    elif startGPS<fileStartGPS and fileEndGPS<endGPS:
                        filesToUse.append(file)
                        indecesToUse.append([0,fileDuration])
                        if verbose : print('case3')
                        continue
                    # ---(---[----)---]---
                    elif fileStartGPS<startGPS<fileEndGPS and fileEndGPS<endGPS:
                        filesToUse.append(file)
                        indecesToUse.append([int(startGPS-fileStartGPS)
                                              ,fileDuration])
                        if verbose : print('case4')
                        continue

                if verbose :
                    for f in range(len(filesToUse)):
                        print(filesToUse[f],indecesToUse[f])


                if len(filesToUse)!=0 : 
                    theSet=DataSet.load(masterDirectory+"/"+det+"/"
                                        +filesToUse[0])[indecesToUse[0][0]:indecesToUse[0][1]]
                    for f in range(1,len(filesToUse)):
                        subset=DataSet.load(masterDirectory+"/"+det+"/"
                                            +filesToUse[f])[indecesToUse[f][0]:indecesToUse[f][1]]
                        theSet.add(subset)
                else:
                    raise ValueError("No files passed to be used")
            # Case where we provide a list of dataSets
            elif all(os.path.isfile(masterDirectory+"/"+det+"/"+el) for el in dataSets):

                if len(dataSets)!=0 : 
                    theSet=DataSet.load(masterDirectory+"/"+det+"/"+dataSets[0])
                    for f in range(1,len(dataSets)):
                        subset=DataSet.load(masterDirectory+"/"+det+"/"+dataSets[f])
                        theSet.add(subset)
                else:
                    raise ValueError("No files passed to be used")

        else:
            raise TypeError("dataSets must be either a list of the files"
                            " to insided masterDirectory, or two gps times, "
                            "indicating a range of files")

        dataset_dict[det]=theSet

    return dataset_dict, duration, fs, windowSize

def check_lags(lags=None,**kwargs):


    if lags == None and 'lags' in kwargs:
        lags = kwarg['lags']
    elif lags == None and 'lags' not in kwargs:
        lags = 1
        

    if not (isinstance(lags,int) and lags >= 0):
        raise ValueError("lags must be an integer greater or equal to zero.")
        
    return(lags)


def check_includeZeroLag(includeZeroLag=None,**kwargs):

    if includeZeroLag == None and 'includeZeroLag' in kwargs:
        includeZeroLag = kwarg['includeZeroLag']
    elif includeZeroLag == None and 'includeZeroLag' not in kwargs:
        includeZeroLag = False
        

    if not isinstance(includeZeroLag,bool):
        raise TypeError("includeZeroLag must be a boolean value True or False.")
        
    return(includeZeroLag)


def check_mass(mass):
    if not isinstance(mass,(int,float)):
        raise TypeError('mass value must be a float or intenger')
    
    if not mass > 0:
        raise ValueError("mass value must be possive")

def check_massRange(massRange):
    
    if isinstance(massRange,(list,tuple)) and len(massRange)==2:
        for mass in massRange:
            check_mass(mass)
    else:
        raise TypeError("massRange must be a tuple or list with size 2, "
                        "indicating a range of masses")
        
def check_validDirectory(path,default=None):
    if path==None:
        return default
    
    if not os.path.isdir(path):
        raise FileNotFoundError(path, " is not found.")
    if path[-1]!="/":
        path=path+"/"
    return path

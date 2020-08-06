import numpy as np
import matplotlib.pyplot as plt
from math import ceil 
from gwpy.timeseries import TimeSeries
from gwpy.io.kerberos import kinit
from gwpy.segments import DataQualityFlag
from gwpy.segments import Segment
from gwpy.time import to_gps
from gwpy.time import from_gps
from dqsegdb2.query import query_segments
import os
import time
import pandas as pd

def nullpath():
    pwd=os.getcwd()
    if 'MLy_Workbench' in pwd:
        null_path=pwd.split('MLy_Workbench')[0]+'MLy_Workbench'
    elif 'EMILY' in pwd:
        null_path=pwd.split('EMILY')[0]+'EMILY'
    else:
        null_path=''
        print('Warning: null_path is empty, you should run import mly, CreateMLyWorkbench()'
              +' to create a workbench or specify null_path value here to avoid FileNotFound errors.')
    return(null_path)

# Creates a list of names of the files inside a folder   

def dirlist(filename,exclude=None):                         
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

# This function creates a list of indeses as begining for every instantiation of
# noise generated. This makes the lag method happen.
def index_combinations(detectors
                       ,lags
                       ,length
                       ,fs
                       ,size
                       ,start_from_sec=0):
    
    indexes={}

    if lags==1:
        for det in detectors:
            indexes[det]=np.arange(start_from_sec*fs,start_from_sec*fs
                                   +size*length*fs, length*fs)
            
    elif lags>=len(detectors):
        
        batches=int(ceil(size/(lags*(lags-1))))
        
        for det in detectors:
            indexes[det]=np.zeros(lags*(lags-1)*batches)

        d=np.int_(range(lags))

        if len(detectors)==1:
            indexes[detectors[0]]=np.arange(start_from_sec*length*fs
                       ,(start_from_sec+size)*length*fs, length*fs)



        elif len(detectors)==2:
            for b in range(0,batches):
                for j in range(1,lags):
                    indexes[detectors[0]][(b*(lags-1)+j-1)*lags
                    :(b*(lags-1)+j)*lags]=(start_from_sec+(b*lags+d)*length)*fs
                    
                    indexes[detectors[1]][(b*(lags-1)+j-1)*lags
                    :(b*(lags-1)+j)*lags]=(start_from_sec
                                           +(b*lags+np.roll(d,j))*length)*fs
                    
        elif len(detectors)==3:
            for b in range(0,batches):
                for j in range(1,lags):

                    indexes[detectors[0]][(b*(lags-1)+j-1)*lags
                    :(b*(lags-1)+j)*lags]=(start_from_sec+(b*lags+d)*length)*fs
                    
                    indexes[detectors[1]][(b*(lags-1)+j-1)*lags
                    :(b*(lags-1)+j)*lags]=(start_from_sec
                                           +(b*lags+np.roll(d,j))*length)*fs
                    indexes[detectors[2]][(b*(lags-1)+j-1)*lags:(b*(lags
                                        -1)+j)*lags]=(start_from_sec
                                        +(b*lags+np.roll(d,-j))*length)*fs

    for det in detectors:
        indexes[det]=np.int_(indexes[det][:size])

    return(indexes)
    

    
def toCategorical(labels,translation = True,from_mapping=None):

    
    if isinstance(from_mapping,list):
        classes=from_mapping
        for label in labels:
            if label not in classes: 
                raise ValueError(str(label)+' was not included in your mapping')
        classesNumerical=list(i for i in range(len(classes)))
        
    else:
        classes=[]
        classesNumerical=[]
        num=0
        for label in labels:
            if label not in classes: 
                classes.append(label)
                classesNumerical.append(num)
                num+=1
                
    translationDict={}
    classesCategorical = []
    for i in range(len(classes)):
        categorical = len(classes)*[0]
        categorical[i]=1
        classesCategorical.append(categorical)
        translationDict[classes[i]] = categorical
    labelsCategorical=[]
    for label in labels:
        labelsCategorical.append(translationDict[label])
    labelsCategorical=np.asarray(labelsCategorical)
    if translation==True:
        return(labelsCategorical,translationDict)
    else:
        return(labelsCategorical)

def fromCategorical(label,model=None,mapping=None,column=False):
    if mapping!=None:
        dictionary = mapping
    elif model!=None:
        dictionary = model.translation
    else:
        raise TypeError('You need to provide mapping')
    categorical = list(dictionary.values())
    labels = list(dictionary.keys())

    if column == True:
        return(np.where(np.array(dictionary[label])==1)[0][0])
    else:
        return(dictionary[label])      
    
    
def correlate(x,y,window):
    result=[]
    for i in np.arange(-window,window):
        xroll=np.roll(x,i)
        cor = np.sum((xroll-np.mean(x))*(y-np.mean(y)))/(
            np.sqrt(np.sum((xroll-np.mean(x))**2))*np.sqrt(np.sum((y-np.mean(y))**2)))
        result.append(cor)
    return np.array(result)







def get_ligo_noise(days
                   , fs=1024 
                   , detectors='HLV' 
                   , destinationPath=None 
                   , minDuration=None
                   , key=None):
    
    # LOG IN WITH LIGO CREDENTIALS
    #if key=='vs' : kinit(keytab= '/home/vasileios.skliris.keytab' )
    #!ligo-proxy-init -k
    
    if isinstance(days,str): days = [days]
    if not all(isinstance(d,str) for d in days): 
        raise ValueError('Days must be in the date format ex: 13 July 2017')
        
    if not (isinstance(fs,int) and fs%2==0):
        raise ValueError('Sample frequency must be a valid number miltpiple of 2')
        
    if not ((isinstance(detectors,str) and (det in 'HLV' for det in detectors))
             or (isinstance(detectors,list)) and (det in 'HLV' for det in detectors)):
        raise ValueError("Detectors can be any collection of 'H','L','V' in string or list format")
        
    if destinationPath==None:
        destinationPath='./'
        
    if minDuration==None:
        minDuration = 128
        
    if isinstance(key,str):
        kinit(keytab = key)
        os.system("ligo-proxy-init -k")
        
    for day in days:
        
        
        gps_start = to_gps(day)
        gps_end = gps_start+24*3600     
        GPS = [gps_start, gps_end]
        
        # Looking for segments that data exists

        det_seg_channels = {'H': 'H1:DMT-ANALYSIS_READY:1'
                           ,'L': 'L1:DMT-ANALYSIS_READY:1'
                           ,'V': 'V1:ITF_SCIENCE:1'}
        det_segs=[]
        for det in detectors:
            det_segs.append(query_segments(det_seg_channels[det], GPS[0],GPS[1]))

        # Finding the time segments where all detectors are on observing mode.
        
        sim_seg = det_segs[0]['active']
        for seg in det_segs[1:]:
            sim_seg = sim_seg & seg['active']

        if not sim_seg==[]:

            seg_time=[]      
            sim_time_total=0 
            seg_no=0         

            maxseg=10000
            for seg in sim_seg:
                if seg[1]-seg[0] >maxseg:
                    for r in range(int((seg[1]-seg[0])//maxseg)):
                        seg_time.append([seg[0]+r*maxseg,seg[0]+(r+1)*maxseg])
                        sim_time_total+=maxseg
                        seg_no+=1
                    seg_time.append([seg[0]+maxseg*((seg[1]-seg[0])//maxseg),seg[1]])
                    sim_time_total+=(seg[1]-(seg[0]+maxseg*((seg[1]-seg[0])//maxseg)))
                    seg_no+=1
                elif seg[1]-seg[0] < minDuration:
                    continue
                else:
                    seg_time.append([seg[0],seg[1]])
                    sim_time_total+=(seg[1]-seg[0])
                    seg_no+=1
            print('There are '+str(seg_no)+' segments of simultaneous GPS time, with total duration of \n'
                  +str(sim_time_total)+' seconds  =  '+str(sim_time_total/3600)+' hours')
            print('More specifically:')
            for i in range(0,len(seg_time)):
                print('Segment No:'+str(i)+' Time: '+str(seg_time[i][0])
                      +' to '+str(seg_time[i][1])+' Duration: '+str(seg_time[i][1]-seg_time[i][0]))    

            # Creating director name acording to date

            year=str(from_gps(gps_start).year)
            month=str(from_gps(gps_start).month)
            day_=str(from_gps(gps_start).day)

            if len(month)==1: month='0'+month
            if len(day_)==1: day_='0'+day_

            filename = year + month + day_

            if os.path.isdir(destinationPath+filename):
                pass
                #raise NameError('Directory '+filename+' already exists')
            else:
                os.mkdir(destinationPath+filename)
                for det in detectors:
                    os.mkdir(destinationPath+filename+'/'+det)
            print('File '+destinationPath+filename+' has been created.')


            # Getting the data and saving them downsampled to the desired sample frequency

            det_channels = {'H': 'H1:GDS-CALIB_STRAIN'
                           ,'L': 'L1:GDS-CALIB_STRAIN'
                           ,'V': 'V1:Hrec_hoft_16384Hz'}
            det_frametypes = {'H': 'H1_HOFT_C00'
                             ,'L': 'L1_HOFT_C00'
                             ,'V': 'V1Online'}
            for segment in range(seg_no):

                gpsStart = seg_time[segment][0]
                gpsEnd   = seg_time[segment][1]
                
                if gpsEnd-gpsStart >= minDuration:

                    for det in detectors:
                        
                        try:
                            data=TimeSeries.get( det_channels[det]
                                                 , gpsStart
                                                 , gpsEnd
                                                 , frametype = det_frametypes[det]
                                                 , verbose=True).astype('float64')
                        except:
                            if isinstance(key,str):
                                os.system("ligo-proxy-init -k")
                                try:
                                    data=TimeSeries.get( det_channels[det]
                                                         , gpsStart
                                                         , gpsEnd
                                                         , frametype = det_frametypes[det]
                                                         , verbose=True).astype('float64')
                                except:
                                    break
                            else:
                                break
                        t0=time.time()
                        print(len(data)/(16*fs))
                        data=data.resample(fs)
                        t1=time.time()-t0
                        print(t1)

                        nameref = str('SEG'+str(segment)+'_'+str(int(gpsStart))
                                      +'_'+str(int(gpsEnd-gpsStart))+'s.txt')

                        np.savetxt(destinationPath+filename+'/'+det+'/'+nameref,data)
                        print('File '+nameref+' is successfully saved')
                else:
                    print('Duration '+str(int(gpsEnd-gpsStart))+'s is below minimum duration')
        else:
            print('No simultaneous GPS time for '+day+', sorry ...')



            


def get_ligo_glitches(dayStart                   
                   , outputDuration
                   , dayEnd=None
                   , fs=1024 
                   , destinationPath=None 
                   , key=None
                   , **kwargs):
    
    # LOG IN WITH LIGO CREDENTIALS
    #if key=='vs' : kinit(keytab= '/home/vasileios.skliris.keytab' )
    #!ligo-proxy-init -k
    
    detectors=['H1','L1','V1']
    runs={ 'O1':[1126623617,1136649617] 
          ,'O2':[1164556817,1187733618]
          ,'O3a':[1238166018,1253977218]
          ,'O3b':[1256655618,1269363618]}
    
    if dayStart in list(runs.keys()):
        run=dayStart
        theRun=dayStart
        dayStart=runs[run][0]
        dayEnd=runs[run][1]
    else: 
        theRun=None

        
    if isinstance(dayStart,str): dayStart=to_gps(dayStart)
        
    if dayEnd==None: dayEnd=dayStart+3600*24
    if isinstance(dayEnd,str): dayEnd=to_gps(dayEnd)
        
                
    if not (isinstance(fs,int) and fs%2==0):
        raise ValueError('Sample frequency must be a valid number miltpiple of 2')
        
    if destinationPath==None:
        destinationPath='./'

    if isinstance(key,str):
        kinit(keytab= key)
        os.system("ligo-proxy-init -k")

    # load glitch dataFrame
    print(dayStart,dayEnd)
    for run in list(runs.keys()):
        if (runs[run][0]<=dayStart<=runs[run][1] 
            and runs[run][0]<=dayEnd<=runs[run][1]):
            theRun=run
    if theRun==None:
        raise ValueError('You can only get glitches in one Run at a time.\n'
                             +'dayStart and dayEnd are from different runs')
    
    glitchFrame=pd.read_csv('glitch_'+theRun+'_H1.csv')
    for d in range(1,3):
        glD=pd.read_csv('glitch_'+theRun+'_'+detectors[d]+'.csv')
        glitchFrame=glitchFrame.append(glD,ignore_index=True)
    
    glitchFrame=glitchFrame[glitchFrame['GPStime']>=dayStart]
    glitchFrame=glitchFrame[glitchFrame['GPStime']<=dayEnd]
    
    glitchFrame['maxFreq']=glitchFrame['centralFreq']+glitchFrame['bandwidth']/2

    print(kwargs)
    for option in kwargs:
        if option not in list(glitchFrame.columns):
            raise KeyError(option+' is not a valid option')
        if (isinstance(kwargs[option],tuple)
            and len(kwargs[option])==2 
            and all(isinstance(op,(int,float)) for op in kwargs[option])):
            glitchFrame=glitchFrame[kwargs[option][0]<=glitchFrame[option]]
            glitchFrame=glitchFrame[kwargs[option][1]>=glitchFrame[option]]
        elif isinstance(kwargs[option],list):
            glitchFrame=glitchFrame[glitchFrame[option] in kwargs[option]]
        
        elif isinstance(kwargs[option],(str,int,float)):
            glitchFrame=glitchFrame[glitchFrame[option]==kwargs[option]]
        

    
    glitchFrame=glitchFrame.sort_values(by='GPStime')
    gl=glitchFrame.reset_index(drop=True)
    
    window=16*outputDuration

    det_channels = {'H1': 'H1:GDS-CALIB_STRAIN'
                   ,'L1': 'L1:GDS-CALIB_STRAIN'
                   ,'V1': 'V1:Hrec_hoft_16384Hz'}
    det_frametypes = {'H1': 'H1_HOFT_C00'
                     ,'L1': 'L1_HOFT_C00'
                     ,'V1': 'V1Online'}
    
    gl['filename']=gl.shape[0]*[None]
    for i in range(gl.shape[0]):
        
        ifo=gl['ifo'].iloc[i]
        GPS=[ gl.loc[[i],['GPStime']].values[0][0]-window/2
             ,gl.loc[[i],['GPStime']].values[0][0]+window/2]    
        
        fileName=gl.loc[[i],['id']].values[0][0]+'_'+str(int(gl.loc[[i],['GPStime']].values[0][0]))+'_1s.txt'
        
        if os.path.isfile(destinationPath+fileName):
            gl.loc[[i],['filename']]= fileName
            continue
        try:
            data=TimeSeries.get(det_channels[ifo]
                                , GPS[0],GPS[1]
                                , frametype=det_frametypes[ifo]
                                ,verbose=False).astype('float64')
            data=data.resample(fs)
            print(len(data))
            D=np.array(data.whiten(1,0.5)[int(fs*(window-outputDuration)/2): int(fs*(window+outputDuration)/2)])
            D=D-np.mean(D)
            np.savetxt(destinationPath+fileName,D)
            gl.loc[[i],['filename']]= fileName
            
            print(len(D))
        except:
            try:
                os.system("ligo-proxy-init -k")
                data=TimeSeries.get(det_channels[ifo]
                                    , GPS[0],GPS[1]
                                    , frametype=det_frametypes[ifo]
                                    ,verbose=False).astype('float64')
                data=data.resample(fs)
                print(len(data))
                D=np.array(data.whiten(1,0.5)[int(fs*(window-outputDuration)/2): int(fs*(window+outputDuration)/2)])
                D=D-np.mean(D)
                gl.loc[[i],['filename']]= fileName
                np.savetxt(destinationPath+fileName,D)
                print(len(D))
            except:
                print('skipped gitch')
                gl.loc[[i],['filename']]= None
        
    gl.to_pickle(destinationPath+'index.pkl')

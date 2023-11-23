# import pandas as pd
import numpy as np
# import pickle
import os
# import sys
# import time
# import random
# import copy
# from math import ceil
import subprocess
from dqsegdb2.query import query_segments
from gwpy.io.kerberos import kinit

from .simulateddetectornoise import *
from .tools import dirlist,  fromCategorical, correlate,internalLags,circularTimeSlides
from .datatools import DataPod, DataSet, generator

from .exceptions import *
from gwpy.time import to_gps,from_gps
from gwpy.segments import DataQualityFlag
from gwpy.segments import Segment,SegmentList

# from gwpy.io.kerberos import kinit

# from gwpy.timeseries import TimeSeries
# import matplotlib.pyplot as plt
# from matplotlib.mlab import psd

# from tensorflow.keras.models import load_model, Sequential, Model
from pycondor import Job, Dagman

which_python = subprocess.check_output('which python', shell=True, text=True)

def getSegments(
              duration 
             ,fs
             ,detectors
             ,dates = None
             ,windowSize = None
             ,observingFlags = None
             ,excludedFlags = None
             ,maxSegmentSize = None):
    
    # Input checks
    check_duration(duration)
    check_fs(fs)
    
    detectors = check_detectors(detectors)
    gps_start, gps_end = check_dates(dates)
    windowSize = check_windowSize(windowSize, duration)
    observingFlags = check_observingFlags(observingFlags)
    excludedFlags = check_excludedFlags(excludedFlags)
    maxSegmentSize = check_maxSegmentSize(maxSegmentSize, duration, windowSize)
    

              
    observingSegments=[]
    for d in range(len(detectors)):
        main_seg=query_segments(observingFlags[0][detectors[d]] ,gps_start,gps_end)['active']
        for obs in observingFlags[1:]:
            try:
                main_seg = main_seg & query_segments(obs[0][detectors[d]]
                                                     ,gps_start,gps_end)['active']
            except KeyError:
                pass
            except:
                raise

            for exc in excludedFlags:
                try:
                    main_seg = main_seg & ~query_segments(exc[detectors[d]]
                                                          ,gps_start,gps_end)['active']
                except KeyError:
                     pass
                except:
                    raise

        observingSegments.append(main_seg)

    coinsidentSegments = observingSegments[0]
    for seg in observingSegments[1:]:
        coinsidentSegments = coinsidentSegments & seg
    
    dissectedSegments=[]
    
    for seg in coinsidentSegments:
        segsizeUtilisation = seg[1]-seg[0]
        print('     ',segsizeUtilisation,int(segsizeUtilisation/maxSegmentSize))
        

        if segsizeUtilisation >= windowSize:
            k=0
            while segsizeUtilisation > maxSegmentSize:
                dissectedSegments.append(Segment(seg[0]+k*maxSegmentSize
                                          ,seg[0]+(k+1)*maxSegmentSize+windowSize-duration))
                segsizeUtilisation -= maxSegmentSize
                k+=1
                print(0,dissectedSegments[-1][1]-dissectedSegments[-1][0])
            
            if segsizeUtilisation >= windowSize and k>0:
                dissectedSegments.append(Segment(seg[0]+k*maxSegmentSize,seg[1]))
                print(1,dissectedSegments[-1][1]-dissectedSegments[-1][0])

            elif segsizeUtilisation >= windowSize and k==0:
                dissectedSegments.append(Segment(seg[0],seg[1]))
                print(2,dissectedSegments[-1][1]-dissectedSegments[-1][0])

        else:
            continue
    return dissectedSegments

#getSegments(1,1024,'HLV',dates=['1 Aug 2017','2 Aug 2017'])


def createFileSysem(duration 
             ,fs
             ,detectors
             ,labels = None
             ,dates = None
             ,windowSize = 16
             ,backgroundType=None
             ,masterDirectory=None
             ,startingPoint = None
             ,frames=None 
             ,channels=None
             ,**kwargs):
             
    
    # Input checks
    check_duration(duration)
    check_fs(fs)
    detectors = check_detectors(detectors)
    labels = check_labels(labels)
    windowSize = check_windowSize(windowSize, duration)
    masterDirectory = check_masterDirectory_createFS(masterDirectory, detectors)
    backgroundType = check_backgroundType(backgroundType)
    frames = check_frames(frames,duration)
    channels = check_channels(channels,duration)
    
    accounting_group_user = check_kw_accounting_group_user(**kwargs)
    accounting_group = check_kw_accounting_group(**kwargs)

        
    error = masterDirectory+'condor/error'
    output = masterDirectory+'condor/output'
    log = masterDirectory+'condor/log'
    submit = masterDirectory+'condor/submit'

    if 'dagman_name' in kwargs:
        dagman_name = kwargs['dagman_name']
    else: 
        dagman_name = 'createFileSystemDagman'
    dagman = Dagman(name = dagman_name,
            submit=submit)
    job_list=[]

    kwstr=""
    for k in kwargs:
        if k not in ['accounting_group_user','accounting_group','dagman_name']:
            kwstr+=(","+k+"="+str(kwargs[k]))       
        
    if backgroundType=='real':
        
        dates = check_dates(dates)


        segments = getSegments(duration = duration
                               ,fs = fs
                               ,detectors = detectors
                               ,dates = dates
                               ,windowSize = windowSize
                               ,observingFlags = check_observingFlags(**kwargs)
                               ,excludedFlags = check_excludedFlags(**kwargs)
                               ,maxSegmentSize = check_maxSegmentSize(None,duration,windowSize,**kwargs)
                              )
        
    elif backgroundType=='optimal':
        
        if not ('numberOfSegments' in kwargs and isinstance(kwargs['numberOfSegments'],int)):
            raise ValueError("For optimal noise use, the numberOfSegments value is"
                             " needed")
        
        segments=np.arange(kwargs['numberOfSegments'])

    for i in range(len(segments)):

        if backgroundType=='real':

            size = int(segments[i][1]-segments[i][0]-(windowSize-duration))
            segmentFileName = str(int(segments[i][0]))+'-'+str(int(segments[i][1]))+'_'+str(size)
            
        elif backgroundType=='optimal':
            size = kwargs['maxSegmentSize']
            segmentFileName = 'optimalNoise-No'+str(i+1)+'_'+str(size)
            
        with open(masterDirectory+'script_'+segmentFileName+'.py','w') as f:
            f.write('#!'+which_python+'\n')
            f.write('import os\n')
            f.write('os.environ["GWDATAFIND_SERVER"]="datafind.ldas.cit:80"\n')

            f.write('from mly.datatools import DataPod, DataSet, generator\n\n')
            
            f.write("import time\n\n")
            f.write("t0=time.time()\n")
            

            if backgroundType == 'optimal':
                for det in detectors:
                    comand=( "SET = generator(\n"
                             +24*" "+"duration = "+str(duration)+"\n"
                             +24*" "+",fs = "+str(fs)+"\n"
                             +24*" "+",size = "+str(size)+"\n"
                             +24*" "+",detectors = '"+det+"'\n"
                             +24*" "+",labels = "+str(labels)+"\n"
                             +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                             +24*" "+",windowSize ="+str(windowSize)+"\n"
                             +24*" "+",name = '"+segmentFileName+"'\n"
                             +24*" "+",savePath ='"+masterDirectory+det+"'"+kwstr+")\n")
            
                    f.write(comand+'\n\n')
                f.write("print(time.time()-t0)\n")

            else:
                #f.write("sys.path.append('"+date_list_path[:-1]+"')\n")
                for det in detectors:
                    comand=( "SET = generator(\n"
                             +24*" "+"duration = "+str(duration)+"\n"
                             +24*" "+",fs = "+str(fs)+"\n"
                             +24*" "+",size = "+str(size)+"\n"
                             +24*" "+",detectors = '"+det+"'\n"
                             +24*" "+",labels = "+str(labels)+"\n"
                             +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                             +24*" "+",noiseSourceFile = "+str([[segments[i][0],segments[i][1]]])+"\n"
                             +24*" "+",windowSize ="+str(windowSize)+"\n"
                             +24*" "+",startingPoint = "+str(startingPoint)+"\n"
                             +24*" "+",name = '"+segmentFileName+"'\n"
                             +24*" "+",savePath ='"+masterDirectory+det+"/'\n"
                             +24*" "+",frames ="+str(frames)+"\n"
                             +24*" "+",channels ="+str(channels)+"\n"
                             +24*" "+",shuffle="+str(False)+kwstr+")\n")
            
                    f.write(comand+'\n\n')
                f.write("print(time.time()-t0)\n")

        os.system('chmod 777 '+masterDirectory+'script_'+segmentFileName+'.py' )
        job = Job(name=segmentFileName
                  ,executable=masterDirectory+'script_'+segmentFileName+'.py' 
                   ,retry = 5
                   ,submit=submit
                   ,error=error
                   ,output=output
                   ,log=log
                   ,getenv=True
                   ,dag=dagman
                   ,request_disk = '50M'
                   ,extra_lines=["accounting_group_user="+accounting_group_user
                                 ,"accounting_group="+accounting_group] )

        job_list.append(job)

    dagman.build_submit()

    



# createFileSysem(duration=1
#                  ,fs=1024
#                  ,detectors='HLV'
#                  ,labels = None
#                  ,windowSize = 16
#                  ,backgroundType='optimal'
#                  ,masterDirectory="/home/vasileios.skliris/masterdir_optimal"
#                  ,numberOfSegments=100
#                  ,maxSegmentSize=1024)                           

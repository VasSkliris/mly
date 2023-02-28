import tensorflow as tf
import pandas as pd
# Path sourcing and access to the ligo real data
import sys
import os
from mly.datatools import DataPod, DataSet
from mly.validators import *
from mly.checkingFunctions import *
from mly.waveforms import cbc

from mly.plugins import *
# Python packages that we will need
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model


# tf.autograph.set_verbosity(
#     0, alsologtostdout=False
# )




def assembleDataSet( masterDirectory
                    ,dataSets
                    , detectors
                    , batches = 1
                    , batchNumber=1
                    , lags=1
                    , includeZeroLag=False):
    
    """A function to assemble an file system as created by the
    mly.tools.createFileSystem function, into a DataSet object
    to appropriate for FAR calculations. It uses streams of already
    processed data and creates unique combinations with time shifts.
    Those time shifted data are then organised into a DataSet object.
    
    Parameters
    ----------
    
    masterDirectory : str 
        The path to the head directory of the file system to be used.
        
    dataSets : list of paths / gps time interval
        This parameter determines which files of that file system to use.
        You can provide a list of files that exist in each of the detector
        directories. Or you can give a list of gps times [gps start, gps end]
        and the function will fetch the appropriate slices of the files that
        include those times.
        
    detectors : str/list
        The list of detectors to be used. You should use the initials of 
        each detector in a string or list format ex. 'HLV' or ['H','L','V'].
    
    batches : int (optional)
        Indicates the number of parallel processes to be run in case you 
        want to use parallel processing. Default is 1. If you use that,
        the script will understand that creation of the dataset is split
        into that number of batches, and will use batchNumber to understand
        which batch is the current one.
        
    batchNumber : int (optional)
        The loop parameter for the parallel processing. Defult is 1.
        
    lags : int
        The number of time shifts used on the stream of data provided. This
        number affects the size of the resulted DataSet. If the size of 
        the dataSet streams are N, the resulted dataSet will have size N*(lags-1).
        This parameter defaults to 1.
        
    includeZeroLag : bool 
        Option if we want to use instances of data that were taken at the same
        gps time. Usually this is not used in FAR calculation so the default is
        False. if true the the final size will be N*lags.
        
    Returns
    -------
    
    dataset : mly.datatools.DataSet 
        The final dataset with all the combination requested.
        
    
    """
    
    t0=time.time()
    
    masterDirectory=check_masterDirectory_verifyFS(masterDirectory,detectors)
    dataset_dict, duration, fs, windowSize = check_dataSets_asinput(dataSets,detectors, masterDirectory,verbose=False)

    lags=check_lags(lags)
    includeZeroLag=check_includeZeroLag(includeZeroLag)
    
    
    INDEX=internalLags(detectors = detectors             # The initials of the detectors you are going 
                     ,duration = 1             # The duration of the instances you use
                     ,size = len(dataset_dict[detectors[0]])            # Size in seconds of the available segment
                     ,fs=1          # Sample frequency
                     ,lags=lags
                     ,includeZeroLag=False) # Includes zero lag by defult !!!  
    
    batchSize= int(len(INDEX[detectors[0]])/batches)
        
    podList=[]
    for b in range(batchNumber*batchSize, (batchNumber+1)*batchSize):

        strain=np.concatenate(list(dataset_dict[det][INDEX[det][b]].strain for det in detectors),axis=0)
        gps_=np.concatenate(list(dataset_dict[det][INDEX[det][b]].gps for det in detectors), axis=0).tolist()

        pod=DataPod(strain, detectors=detectors,fs=fs, gps=gps_)
        pod.addPlugIn(knownPlugIns('correlation_30'))

        podList.append(pod)

    dataSet = DataSet(podList)

    if 'V' not in detectors:
        dataSet.stackDetector(**{ 'duration':duration
                                        ,'fs':fs
                                        ,'detectors' : 'V'
                                        ,'windowSize':windowSize
                                        ,'backgroundType' :'optimal'
                                        ,'PSDm':{'V':32}})

    return dataSet


def testModel(model
              , dataSet
              , restriction=0
              , labels={'type':'noise'}
              , mapping= 2*[{ "noise" : [1, 0],"signal": [0, 1]}]):
    
 
    t0=time.time()
    
    models, trainedModels = check_models(model, returnTrainedModels=True)

    result_list=[]
    scores_collection=[]
    
    columns=[]
    for m in range(len(trainedModels)):

        columns.append(fromCategorical(labels['type'],mapping=mapping[m],column=True))
              
    for m in range(len(trainedModels)):
        dataList=[]
        input_shape=trainedModels[m].input_shape
        if isinstance(input_shape,tuple): input_shape=[input_shape]
        for i in range(len(models[1][m])):
            print(input_shape[i],models[1][m][i])
            #print(dataSet[0].__getattribute__(models[1][m][i]).shape)
            dataList.append(dataSet.exportData(models[1][m][i],shape=input_shape[i]))

        if len(dataList)==1: dataList=dataList[0]
        
        
        scores = 1.0 - trainedModels[m].predict(dataList, batch_size=1, verbose=0)[:,columns[m]]
        
        scores_collection.append(scores.tolist())

    inferencetime=time.time()
    print("INFERENCE TIME:", inferencetime-t0," ")

    gps_times=dataSet.exportGPS()

    if len(scores_collection)==1:
        scores_collection=np.expand_dims(np.array(scores_collection),0)
    else:
        scores_collection=np.array(scores_collection)
    scores_collection=np.transpose(scores_collection)

    #print(scores_collection.shape,np.array(gps_times).shape)
    result=np.hstack((scores_collection,np.array(gps_times)))

    result_pd = pd.DataFrame(result ,columns = list('scores'+str(m+1) for m in range(len(trainedModels)))
                             +list('GPS'+str(det) for det in dataSet[0].detectors))

    for m in range(len(trainedModels)):
        if m==0: 
            result_pd['total']=result_pd['scores'+str(m+1)]
        else:
            result_pd['total']=result_pd['total']*result_pd['scores'+str(m+1)]

    result_pd = result_pd.sort_values(by=['total'],ascending = False)

    if isinstance(restriction,(int,float)):
        result_pd=result_pd[result_pd['total']>=restriction]


    finalisationtime=time.time()
    print("RESULT FRAMING TIME:", finalisationtime-inferencetime," ")

    print("\n\n ----")

    return(result_pd)
              
              




import pandas as pd
import numpy as np
import pickle
import os
import sys
import time
import random
import copy
from math import ceil
from dqsegdb2.query import query_segments
from gwpy.io.kerberos import kinit

from .simulateddetectornoise import *
from .tools import dirlist,  fromCategorical, correlate,internalLags,circularTimeSlides
from .datatools import DataPod, DataSet , generator, stackDetector
from gwpy.time import to_gps,from_gps
from gwpy.segments import DataQualityFlag
from gwpy.segments import Segment,SegmentList

from gwpy.io.kerberos import kinit

from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from matplotlib.mlab import psd

from tensorflow.keras.models import load_model, Sequential, Model
from pycondor import Job, Dagman


################################################################################
################################################################################
class Validator:
    
    def accuracy(models
                       ,duration
                       ,fs
                       ,size
                       ,detectors 
                       ,injection_source = None
                       ,labels = {'type':'signal'}
                       ,backgroundType = None
                       ,injectionSNR = None
                       ,noiseSourceFile = None  
                       ,windowSize = None #(32)            
                       ,timeSlides = None #(1)
                       ,startingPoint= None #(32)
                       ,name = None
                       ,savePath = None
                       ,single = False  # Making single detector injections as glitch
                       ,injectionCrop = 0  # Allows to crop part of the injection when you move the injection arroud, 0 is no 1 is maximum means 100% cropping allowed. The cropping will be a random displacement from zero to parto of duration.
                       ,frames=None 
                       ,channels=None
                       ,maxDuration=None
                       ,disposition=None
                       ,differentSignals=False   # In case we want to put different injection to every detector.
                       ,plugins=None
                       ,mapping=None
                       ,**kwargs):

        # ---------------------------------------------------------------------------------------- #    
        # --- model ------------------------------------------------------------------------------ #
        # 
        # This first input has a complicated format in the rare case of trying
        # to test two models in parallel but with different subset of the data as input.
        #
        # Case of one model as it was before
        if not isinstance(models,list):
            models=[[models],[None]]
        # Case where all models have all data to use
        if isinstance(models,list) and not all(isinstance(m,list) for m in models):
            models=[models,len(models)*[None]]
        # Case where index is not given for all models.
        if len(models[0])!=len(models[1]):
            raise ValueError('You have to define input index for all maodels')
        # Case somebody doesn't put the right amount of indexes for the data inputs. 

        if not (isinstance(models,list) and all(isinstance(m,list) for m in models)):
            raise TypeError('models have to be a list of two sublists. '
                            +'First list has the models and the second has the'
                            +' indexes of the data each one uses following the order strain, extra1, extra2...'
                            +'[model1,model2,model3],[[0,1],[0,2],[2]]')

        # models[0] becomes the trained models list
        trained_models=[]
        for model in models[0]:  
            if isinstance(model,str):
                if os.path.isfile(model):    
                    trained_models.append(load_model(model))
                else:
                    raise FileNotFoundError("No model file in "+model)
            else:
                trained_models.append(model) 

        # ---------------------------------------------------------------------------------------- #    
        # --- mappings --------------------------------------------------------------------------- #
        # 
        # Mappings are a way to make sure the model has the same translation for the
        # labels as we have. All models trained in mly will have a mapping defined
        # during the data formating in the model training.
        

        if len(trained_models)==1 and isinstance(mapping,dict):
            mapping=[mapping]
        elif len(trained_models)!=1 and isinstance(mapping,dict):
            mapping=len(trained_models)*[mapping]
        if isinstance(mapping,list) and all(isinstance(m,dict) for m in mapping):
            pass
        else:
            
            raise TypeError('Mappings have to be a list of dictionaries for each model.')

        columns=[]
        for m in range(len(trained_models)):
            columns.append(fromCategorical(labels['type'],mapping=mapping[m],column=True))

        # ---------------------------------------------------------------------------------------- #    
        # --- injectionSNR ----------------------------------------------------------------------- #

        if isinstance(injectionSNR,list): 
            snrInList=True
            snrs=injectionSNR
        else:
            snrInList=False

        # ---------------------------------------------------------------------------------------- #    
        # --- injectionHRSS ---------------------------------------------------------------------- #
        
        if 'injectionHRSS' in kwargs:
            injectionHRSS = kwargs['injectionHRSS']
            
        else: 
            injectionHRSS=None
            
        if isinstance(injectionHRSS,list): 
            hrssInList=True
            hrsss=injectionHRSS
        else:
            hrssInList=False

        # ---------------------------------------------------------------------------------------- #    
        result={}            
        looper=[]
        if snrInList==True and hrssInList==True:
            raise ValueError('You cannot loop through two values. Do seperate tests')
        elif snrInList==True and hrssInList==False:
            for snr in snrs:
                looper.append(snr)
            result['snrs']=[]
            loopname='snrs'
        elif snrInList==False and hrssInList==True:
            for h in hrsss:
                looper.append(h)
            result['hrss']=[]
            loopname='hrss'
        else:
            raise ValueError('You need to choose a looper value, injectionSNR or injectionHRSS')


        for val in looper:
            if hrssInList==True:

                kwargs['injectionHRSS']=val
            else: 
                injectionSNR=val

            DATA= generator(duration = duration
                                   ,fs = fs
                                   ,size = size
                                   ,detectors = detectors
                                   ,injection_source = injection_source
                                   ,labels = labels
                                   ,backgroundType = backgroundType
                                   ,injectionSNR = injectionSNR
                                   ,noiseSourceFile = noiseSourceFile
                                   ,windowSize = windowSize          
                                   ,timeSlides = timeSlides
                                   ,startingPoint = startingPoint
                                   ,single = single
                                   ,injectionCrop = injectionCrop
                                   ,frames=frames
                                   ,channels=frames
                                   ,disposition = disposition
                                   ,maxDuration = maxDuration
                                   ,differentSignals = differentSignals
                                   ,plugins=plugins
                                   ,**kwargs)

            
            if 'stackDetectorDict' in kwargs:
                DATA = stackDetector(DATA, **kwargs['stackDetectorDict'])
                
            #random.shuffle(DATA.dataPods)

            #DATA.save(savePath+'/setexample_'+str(val))
            result[loopname].append(val)

            
            for m in range(len(trained_models)):
                dataList=[]
                input_shape=trained_models[m].input_shape
                if isinstance(input_shape,tuple): input_shape=[input_shape]
                for i in range(len(models[1][m])):
                    # print(input_shape[i],models[1][m][i])
                    # print(DATA[0].__getattribute__(models[1][m][i]).shape)
                    dataList.append(DATA.exportData(models[1][m][i],shape=input_shape[i]))

                if len(dataList)==1: dataList=dataList[0]
                scores = trained_models[m](dataList, training=False).numpy()[:,columns[m]]

                if 'scores'+str(m+1) in list(result.keys()):
                    result['scores'+str(m+1)].append(scores.tolist())
                else:
                    result['scores'+str(m+1)]=[scores.tolist()]

        if savePath==None:
            savePath='./'

        if name!=None:
            with open(savePath+name+'.pkl', 'wb') as output:
                pickle.dump(result, output, 4)
        
        return(result)

  

    def falseAlarmTest(models
                       ,duration
                       ,fs
                       ,size
                       ,detectors
                       ,labels = {'type': 'noise'}
                       ,backgroundType = None
                       ,noiseSourceFile = None  
                       ,windowSize = None #(32)            
                       ,timeSlides = None #(1)
                       ,startingPoint= None #(32)
                       ,name = None
                       ,savePath = None
                       ,plugins=None
                       ,mapping=None
                       ,strides=None
                       ,restriction=None
                       ,frames=None 
                       ,channels=None
                       ,**kwargs):    
        
        
        t0=time.time()
        # ---------------------------------------------------------------------------------------- #    
        # --- model ------------------------------------------------------------------------------ #
        # 
        # This first input has a complicated format in the rare case of trying
        # to test two models in parallel but with different subset of the data as input.
        #
        # Case of one model as it was before
        if not isinstance(models,list):
            models=[[models],[None]]
        # Case where all models have all data to use
        if isinstance(models,list) and not all(isinstance(m,list) for m in models):
            models=[models,len(models)*[None]]
        # Case where index is not given for all models.
        if len(models[0])!=len(models[1]):
            raise ValueError('You have to define input index for all maodels')
        # Case somebody doesn't put the right amount of indexes for the data inputs. 

        if not (isinstance(models,list) and all(isinstance(m,list) for m in models)):
            raise TypeError('models have to be a list of two sublists. '
                            +'First list has the models and the second has the'
                            +' indexes of the data each one uses following the order strain, extra1, extra2...'
                            +'[model1,model2,model3],[[0,1],[0,2],[2]]')

        preptime1=time.time()
        print("PREP TIME 1:", preptime1-t0)
        # models[0] becomes the trained models list
        trained_models=[]
        for model in models[0]:  
            if isinstance(model,str):
                if os.path.isfile(model):    
                    trained_models.append(load_model(model))
                else:
                    raise FileNotFoundError("No model file in "+model)
            else:
                trained_models.append(model) 
        preptime2=time.time()
        print("PREP TIME 2:", preptime2-preptime1)
        # ---------------------------------------------------------------------------------------- #    
        # --- mappings --------------------------------------------------------------------------- #
        # 
        # Mappings are a way to make sure the model has the same translation for the
        # labels as we have. All models trained in mly will have a mapping defined
        # during the data formating in the model training.
        

        if len(trained_models)==1 and isinstance(mapping,dict):
            mapping=[mapping]
        elif len(trained_models)!=1 and isinstance(mapping,dict):
            mapping=len(trained_models)*[mapping]
        if isinstance(mapping,list) and all(isinstance(m,dict) for m in mapping):
            pass
        else:
            
            
            raise TypeError('Mappings have to be a list of dictionaries for each model.')

        columns=[]
        for m in range(len(trained_models)):
            columns.append(fromCategorical(labels['type'],mapping=mapping[m],column=True))
        preptime3=time.time()
        print("PREP TIME 3:", preptime3-preptime2)
        # ---------------------------------------------------------------------------------------- #    
        # --- strides ---------------------------------------------------------------------------- #
            
        if strides==None:
            strides=1
        elif not (isinstance(strides,int) and strides <= fs and fs%strides==0):
            raise ValueError('Strides must be a divisible integer of sample frequency')
        
        preptime=time.time()
        print("PREP TIME:", preptime-t0)
        # Using a generator for the data to use for testing
        DATA= generator(duration=duration
                               ,fs =fs
                               ,size=size
                               ,detectors=detectors
                               ,labels=labels
                               ,backgroundType=backgroundType
                               ,injectionSNR = 0
                               ,noiseSourceFile =noiseSourceFile
                               ,windowSize =windowSize       
                               ,timeSlides =timeSlides
                               ,startingPoint=startingPoint
                               ,frames=frames
                               ,channels=channels
                               ,name =name
                               ,plugins=plugins
                               ,**kwargs)   
        t1=time.time()
        print('Time to generation: '+str(t1-t0)+' stride 0')
        t0=time.time()

        for st in range(1,strides):
            print('stride:',st,st*(duration/strides))
            DATA_= generator(duration=duration
                       ,fs =fs
                       ,size=size
                       ,detectors=detectors
                       ,labels=labels
                       ,backgroundType=backgroundType
                       ,injectionSNR = 0
                       ,noiseSourceFile =noiseSourceFile
                       ,windowSize =windowSize       
                       ,timeSlides =timeSlides
                       ,startingPoint=startingPoint+st*(duration/strides)
                       ,frames=frames
                       ,channels=channels
                       ,name =name
                       ,plugins=plugins
                       ,**kwargs)
            
            DATA.add(DATA_)
        

        if 'stackDetectorDict' in kwargs:
            DATA = stackDetector(DATA, **kwargs['stackDetectorDict'])
        
        generationtime=time.time()
        print("GENERATION TIME:", generationtime-preptime)

        result_list=[]
        scores_collection=[]

        for m in range(len(trained_models)):
            dataList=[]
            input_shape=trained_models[m].input_shape
            if isinstance(input_shape,tuple): input_shape=[input_shape]
            for i in range(len(models[1][m])):
                print(input_shape[i],models[1][m][i])
                #print(DATA[0].__getattribute__(models[1][m][i]).shape)
                dataList.append(DATA.exportData(models[1][m][i],shape=input_shape[i]))

            if len(dataList)==1: dataList=dataList[0]
            scores = 1.0 - trained_models[m](dataList, training=False).numpy()[:,columns[m]]
            scores_collection.append(scores.tolist())
            
        inferencetime=time.time()
        print("INFERENCE TIME:", inferencetime-generationtime)
        
        gps_times=DATA.exportGPS()
        
        if len(np.array(scores_collection).shape)==1:
            scores_collection=np.expand_dims(np.array(scores_collection),0)
        else:
            scores_collection=np.array(scores_collection)

        scores_collection=np.transpose(scores_collection)

        result=np.hstack((scores_collection,np.array(gps_times)))

        result_pd = pd.DataFrame(result ,columns = list('scores'+str(m+1) for m in range(len(trained_models)))
                                 +list('GPS'+str(det) for det in DATA[0].detectors))

        for m in range(len(trained_models)):
            if m==0: 
                result_pd['total']=result_pd['scores'+str(m+1)]
            else:
                result_pd['total']=result_pd['total']*result_pd['scores'+str(m+1)]

        result_pd = result_pd.sort_values(by=['total'],ascending = False)
        
        if isinstance(restriction,(int,float)):
            result_pd=result_pd[result_pd['total']>=restriction]
        
        # saving the test size so that it can be retrievable in the finalisation
        result_pd.testSize= size
                
        finalisationtime=time.time()
        print("FINALISATION TIME:", finalisationtime-inferencetime)
        if savePath==None:
            savePath='./'
        
        print(os.getcwd())
        if name!=None:
            with open(savePath+name+'.pkl', 'wb') as output:
                pickle.dump(result_pd, output, 4)
                savingtime=time.time()
                print("SAVING TIME:", finalisationtime-inferencetime)
        print("\n\n ----")
        print("TOTAL TIME:", time.time()-t0)
        
        if 'podreturn' in list(kwargs) and len(DATA)==1:
            return(result_pd,DATA[0])
        else:
            return(result_pd)


#     def glitchTest(models
#                        ,duration
#                        ,fs
#                        ,size
#                        ,glitchSourceFile
#                        ,detectors 
#                        ,labels = {'type':'signal'}
#                        ,backgroundType = None
#                        ,injectionSNR = [0]
#                        ,noiseSourceFile = None  
#                        ,windowSize = None #(32)            
#                        ,timeSlides = None #(1)
#                        ,startingPoint= None #(32)
#                        ,name = None
#                        ,savePath = None
#                        ,single = False  # Making single detector injections as glitch
#                        ,injectionCrop = 0  # Allows to crop part of the injection when you move the injection arroud, 0 is no 1 is maximum means 100% cropping allowed. The cropping will be a random displacement from zero to parto of duration.
#                        ,disposition=None
#                        ,maxDuration=None
#                        ,differentSignals=False   # In case we want to put different injection to every detector.
#                        ,extras=None
#                        ,mapping=None
#                        ,substitute=None):

#         # ---------------------------------------------------------------------------------------- #    
#         # --- model ------------------------------------------------------------------------------ #
#         # 
#         # This first input has a complicated format in the rare case of trying
#         # to test two models in parallel but with different subset of the data as input.
#         #
#         # Case of one model as it was before
#         if not isinstance(models,list):
#             models=[[models],[None]]
#         # Case where all models have all data to use
#         if isinstance(models,list) and not all(isinstance(m,list) for m in models):
#             models=[models,len(models)*[None]]
#         # Case where index is not given for all models.
#         if len(models[0])!=len(models[1]):
#             raise ValueError('You have to define input index for all maodels')
#         # Case somebody doesn't put the right amount of indexes for the data inputs. 

#         if not (isinstance(models,list) and all(isinstance(m,list) for m in models)):
#             raise TypeError('models have to be a list of two sublists. '
#                             +'First list has the models and the second has the'
#                             +' indexes of the data each one uses following the order strain, extra1, extra2...'
#                             +'[model1,model2,model3],[[0,1],[0,2],[2]]')

#         # models[0] becomes the trained models list
#         trained_models=[]
#         for model in models[0]:  
#             if isinstance(model,str):
#                 if os.path.isfile(model):    
#                     trained_models.append(load_model(model))
#                 else:
#                     raise FileNotFoundError("No model file in "+model)
#             else:
#                 trained_models.append(model) 

#         # models[1] becomes the the input inexes of the data 
#         if extras==None:
#             number_of_extras=0
#         else:
#             number_of_extras=len(extras)

#         data_inputs_index=[]
#         for index in models[1]:
#             if index==None :
#                 data_inputs_index.append([k for k in range(number_of_extras+1)])
#             elif all(j<= number_of_extras for j in index):
#                 data_inputs_index.append(index)
#             else:
#                 raise TypeError(str(index)+' is not a valid index')


#         # ---------------------------------------------------------------------------------------- #    
#         # --- mappings --------------------------------------------------------------------------- #

#         # Mappings are a way to make sure the model has the same translation for the
#         # labels as we have. All models trained in mly will have a mapping defined
#         # during the data formating in the model training.

#         if len(trained_models)==1 and isinstance(mapping,dict):
#             mapping=[mapping]
#         elif len(trained_models)!=1 and isinstance(mapping,dict):
#             mapping=len(trained_models)*[mapping]
#         if isinstance(mapping,list) and all(isinstance(m,dict) for m in mapping):
#             pass
#         else:
#             raise TypeError('Mappings have to be a list of dictionaries for each model.')

#         columns=[]
#         for m in range(len(trained_models)):
#             columns.append(fromCategorical(labels['type'],mapping=mapping[m],column=True))


#         # ---------------------------------------------------------------------------------------- #    
#         # --- injectionSNR ----------------------------------------------------------------------- #

#         if isinstance(injectionSNR,list): 
#             snrInList=True
#             snrs=injectionSNR
#         else:
#             snrInList=False

#         # ---------------------------------------------------------------------------------------- #    
#         # --- disposition ------------------------------------------------------------------------ #

#         if isinstance(disposition,list): 
#             dispositionInList=True
#             dispositions=disposition
#         else:
#             dispositionInList=False

#         # ---------------------------------------------------------------------------------------- #    

#         if substitute==None:
#             substitute='R1'   
#             subs=np.arange(len(detectors))
            
#         elif substitute in ['R1','R2','R3']:
#             subs=np.arange(len(detectors))

#         else:
#             if not isinstance(substitute,list):
#                 substitute=[substitute]
#             if all((isinstance(sub,int) and sub<len(detectors)) for sub in substitute):
#                 pass
#             elif all((isinstance(sub,str) and sub in detectors) for sub in substitute):
#                 substitute=list(detectors.index(sub) for sub in substitute)
#             else:
#                 raise ValueError(str(substitute)+' is not a valid form for substitution')


#         DATA=DataSet.generator(duration = duration
#                                ,fs = fs
#                                ,size = size
#                                ,detectors = detectors
#                                #,injection_source = injection_source
#                                ,labels = labels
#                                ,backgroundType = backgroundType
#                                ,injectionSNR = 0
#                                ,noiseSourceFile = noiseSourceFile
#                                ,windowSize = windowSize          
#                                ,timeSlides = timeSlides
#                                ,startingPoint = startingPoint
#                                ,single = single
#                                ,injectionCrop = injectionCrop
#                                ,disposition = disposition
#                                ,maxDuration = maxDuration
#                                ,differentSignals = differentSignals
#                                ,extras = extras)
        
#         random.shuffle(DATA.dataPods)

#         if os.path.isfile(glitchSourceFile+'index.pkl'):
#             with open(glitchSourceFile+'index.pkl','rb') as handle:
#                 gl= pickle.load(handle)  
                
#             _columns=['chisq', 'chisqDof', 'confidence','imgUrl','Q-value','imgUrl','id']
#             for col in _columns:
#                 try:
#                     gl=gl.drop(columns=col)
#                 except:
#                     pass

#             glitch_list={'H': gl[gl['ifo']=='H1']['filename'].tolist(),
#                          'L': gl[gl['ifo']=='L1']['filename'].tolist(),
#                          'V': gl[gl['ifo']=='V1']['filename'].tolist()}

#             random.shuffle(glitch_list['H'])
#             random.shuffle(glitch_list['L'])
#             random.shuffle(glitch_list['V'])
            
#             frames={}
#             if substitute in ['R1','R2','R3']:
#                 for n in range(int(substitute[1])):
#                     frames[n]=gl.iloc[:0]
#             else:
#                 for n in range(len(substitute)):
#                     frames[n]=gl.iloc[:0]

#         else:
#             result=pd.DataFrame([])    
#             glitch_list={'H': dirlist(glitchSourceFile,exclude=['.pkl']),
#                          'L': dirlist(glitchSourceFile,exclude=['.pkl']),
#                          'V': dirlist(glitchSourceFile,exclude=['.pkl'])}

#         filenames={}

#         for d in DATA.dataPods:
#             if substitute in ['R1','R2','R3']:
#                 np.random.shuffle(subs)

#                 _subs=subs[:int(substitute[1])].tolist()
#             else:
#                 _subs=substitute
                
                
#             for n in range(len(_subs)):
#                 if os.path.isfile(glitchSourceFile+'index.pkl'):
#                     g_index=int(np.random.randint(0,len(glitch_list[detectors[_subs[n]]]),1))
#                     frames[n]=frames[n].append(gl[gl['filename']==glitch_list[detectors[_subs[n]]][g_index]])
#                     glitch=np.loadtxt(glitchSourceFile+glitch_list[detectors[_subs[n]]][g_index])
#                 else:
#                     g_index=np.random.randint(0,len(glitch_list[detectors[_subs[n]]]))
#                     glitch=np.loadtxt(glitchSourceFile+glitch_list[detectors[_subs[n]]][g_index])
#                     if detectors[_subs[n]] in list(filenames.keys()):
#                         filenames[detectors[_subs[n]]]=[glitch_list[detectors[_subs[n]]][g_index]]
#                     else:
#                         filenames[detectors[_subs[n]]].append(glitch_list[detectors[_subs[n]]][g_index])
#                 d._strain[_subs[n]]=glitch

#             podCorrelations=[]
#             for i in np.arange(len(d.detectors)):
#                 for j in np.arange(i+1,len(d.detectors)):
#                     window= int((2*6371/300000)*fs)+1
#                     podCorrelations.append(correlate(d.strain[i],d.strain[j],window))  
#             d.metadata['correlation']=np.array(podCorrelations)
            
#         if os.path.isfile(glitchSourceFile+'index.pkl'):
#             for n in list(frames.keys()):
#                 frames[n].columns=list(c+'_'+str(n) for c in frames[n].columns.tolist())
#                 frames[n].reset_index(drop=True, inplace=True)
#             result=pd.concat(list(frames[n] for n in list(frames.keys())), axis=1)
#             newcolumns=result.columns.tolist()
#             index=np.arange(len(newcolumns)).tolist()
#             order=['GPStime','filename','bandwidth','centralFreq'
#                    ,'peakFreq','amplitude','label','maxFreq'
#                    ,'label','duration','ifo','duration','snr']
#             rearanged=[]
#             for o in order:
#                 rearanged+=list(o+'_'+str(n) for n in range(len(frames.keys())))

#             for col in rearanged:
#                 try:
#                     newcolumns.append(newcolumns.pop(newcolumns.index(col)))
#                 except:
#                     pass
#             result=result[newcolumns]
#             result['snr']=np.sqrt(sum(list(result['snr_'+str(n)]**2 for n in list(frames.keys()))))
#         else:
#             for det in list(filenames.keys()):
#                 result['filename_'+det]=filenames[det]

#         for m in range(len(trained_models)):
#             dataList=[]
#             input_shape=trained_models[m].input_shape
#             if isinstance(input_shape,tuple): input_shape=[input_shape]
#             for i in data_inputs_index[m]:
#                 if i==0:
#                     dataList.append(DATA.unloadData(shape=input_shape[i]))
#                 else:
#                     dataList.append(DATA.unloadData(extras = extras[i-1]
#                                 ,shape=input_shape[i]))
#             print(data_inputs_index[m])
#             if len(dataList)==1: dataList=dataList[0]
#             scores =  trained_models[m].predict(dataList, batch_size=1)[:,columns[m]]
#             result['scores'+str(m+1)]=scores.tolist()

#         if savePath==None:
#             savePath='./'


#         if name!=None:
#             result.to_pickle(savePath+name+'.pkl')
#         else:
#             return(result)






def auto_FAR(model
             # DEPRICATED NOT WORKING UNTIL UPDATE
             ,duration 
             ,fs
             ,detectors
             ,size
             ,backgroundType = None
             ,firstDay = None
             ,windowSize = None #(32)            
             ,timeSlides = None #(1)
             ,startingPoint = None
             ,name = None
             ,savePath = None
             ,plugins=None
             ,mapping=None
             ,maxTestSize=None
             ,**kwargs):

    
#     # ---------------------------------------------------------------------------------------- #    
#     # --- model ------------------------------------------------------------------------------ #
        
#     if isinstance(model,str) and os.path.isfile(model):
#         if model[:5]!='/home':
#             cwd = os.getcwd()
#             model = cwd+'/'+model
#         trained_model = load_model(model)
#     else:
#         raise FileNotFoundError("Model file "+model+" was not found.")
        

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
    # --- timeSlides ------------------------------------------------------------------------- #

    if timeSlides == None: timeSlides = 1
    if not (isinstance(timeSlides, int) and timeSlides >=1) :
        raise ValueError('timeSlides has to be an integer equal or bigger than 1')
        
        
    # ---------------------------------------------------------------------------------------- #    
    # --- size ------------------------------------------------------------------------------- #        

    if not (isinstance(size, int) and size > 0):
        raise ValuError("size must be a possitive integer.")

    # ---------------------------------------------------------------------------------------- #    
    # --- backgroundType --------------------------------------------------------------------- #

    if backgroundType == None:
        backgroundType = 'optimal'
    elif not (isinstance(backgroundType,str) 
          and (backgroundType in ['optimal','sudo_real','real'])):
        raise ValueError("backgroundType is a string that can take values : "
                        +"'optimal' | 'sudo_real' | 'real'.")


    # ---------------------------------------------------------------------------------------- #    
    # --- firstDay --------------------------------------------------------------------------- #
    
    if backgroundType in ['sudo_real','real'] and firstDay == None:
        raise ValueError("You have to set the first day of the real data")
    elif backgroundType in ['sudo_real','real'] and isinstance(firstDay,str):
        if '/' in firstDay and os.path.isdir(firstDay):
            sys.path.append(firstDay)
            date_list_path='/'.join(firstDay.split('/')[:-1])+'/'
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

    if windowSize == None: windowSize = int(16*duration)
    if not isinstance(windowSize,int):
        raise ValueError('windowSize needs to be an integral')
    if windowSize < duration :
        raise ValueError('windowSize needs to be bigger than the duration')


    # ---------------------------------------------------------------------------------------- #    
    # --- startingPoint ---------------------------------------------------------------------- #

    if startingPoint == None : startingPoint = windowSize 
    if not (isinstance(startingPoint, int) and startingPoint >=0) :
        raise ValueError('timeSlides has to be an integer')        

    # ---------------------------------------------------------------------------------------- #    
    # --- name ------------------------------------------------------------------------------- #

    if name == None : name = 'untitledFalseAlarmTest'
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')

    # ---------------------------------------------------------------------------------------- #    
    # --- savePath --------------------------------------------------------------------------- #

    if savePath == None : 
        savePath = os.getcwd()
    elif (savePath,str): 
        if not os.path.isdir(savePath) : 
            raise FileNotFoundError('No such file or directory:' +savePath)
    else:
        raise TypeError("Destination Path has to be a string valid path")
    if savePath[:5]!='/home':
        savePath = os.getcwd()+'/'+savePath        
    if savePath[-1] != '/' : savePath=savePath+'/'
    
    # ---------------------------------------------------------------------------------------- #    
    # --- maxTestSize ------------------------------------------------------------------------ #
    
    if maxTestSize == None: maxTestSize=max(10000,timeSlides**2)
    print(maxTestSize)
    if not (isinstance(maxTestSize,int) and maxTestSize >= timeSlides**2):
        raise ValueError("maxTestSize must be at least the timeSlides squared")
        
    #num_of_sets = len(injectionSNR)

    # If noise is optimal it is much more simple
    if backgroundType == 'optimal':

        d={'size' : (size//maxTestSize)*[maxTestSize]
           , 'start_point' : (size//maxTestSize)*[windowSize]
           , 'name' : list(name+'_No'+str(i)+'_'
                           +str(maxTestSize) for i in range((size//maxTestSize)))}
        if size%maxTestSize !=0:
            d['size'].append(size%maxTestSize)
            d['start_point'].append(windowSize)
            d['name'].append(name+'_No'+str(size//maxTestSize+1)+'_'
                           +str(size%maxTestSize))
        print('These are the details of the datasets to be generated: \n')
        for i in range(len(d['size'])):
            print(d['size'][i], d['start_point'][i] ,d['name'][i])
                           
    # If noise is using real noise segments it is complicated   
    else:

        # Generating a list with all the available dates in the ligo_data folder
        date_list=dirlist(date_list_path)

        # The first day is the date we want to start using data.
        # Calculating the index of the initial date
        date=date_list[date_list.index(firstDay)]
        # All dates have an index in the list of dates following chronological order
        date_counter=date_list.index(firstDay)             

        # The following while loop checks and stacks durations of the data in date files. In this way
        # we note which of them are we gonna need for the generation.
        test_total = 0

        # To initialise the generators, we will first create the code and the 
        # names they will have. This will create the commands that generate
        # all the segments needed.

        size_list=[]            # Sizes for each generation of noise 
        starting_point_list=[]  # Starting points for each generation of noise(s)
        seg_list=[]           # Segment names for each generation of noise
        name_list=[]            # List with the name of the set to be generated
        timeSlide_list=[]
        set_num=0 
        
        while size > test_total:
            date_counter+=1
            segments=dirlist( date_list_path+date+'/'+detectors[0])
            print(date)
            for seg in segments:
                print(' '+30*'-'+' '+str(100*test_total/size)[:5]+"%")
                gps = seg.split('_')[-2]
                dur = seg.split('_')[-1][:-4]
                
                dur = (int(dur) 
                       -(windowSize-duration) # the part used for psd (only in the beggining)
                        -2*windowSize) # ignoring the edges
    
                if dur <=0:
                    print('    '+seg+' is too small')
                    continue
                
                utilised_dur = timeSlides*duration*(dur//(timeSlides*duration))
                
                utilised_dur_left=dur-timeSlides*duration*(dur//(timeSlides*duration))
                                                           
                local_starting_point=windowSize
                print(' ',seg,utilised_dur,utilised_dur_left,local_starting_point, dur)
                # Step 1: Starting with a segment we try not to create huge files.
                # For that reason we use a max_size for one test file to be the closest
                # multiple of timeslides*(timeslides- 1 or 2) to 10000 that is still less
                # than 10000. After utilising as much as possible of those big files we go
                # to the next step.

                if timeSlides==1: 
                    max_size = maxTestSize
                    max_dur = maxTestSize*duration
                if timeSlides%2 == 0: 
                    max_size = timeSlides*(timeSlides-2)*(maxTestSize//(timeSlides*(timeSlides-2)))
                    max_dur = timeSlides*duration*(max_size//(timeSlides*(timeSlides-2)))
                if timeSlides%2 != 0 and timeSlides !=1 :
                    max_size = timeSlides*(timeSlides-1)*(maxTestSize//(timeSlides*(timeSlides-1)))
                    max_dur = timeSlides*duration*(max_size//(timeSlides*(timeSlides-1)))
                

                while utilised_dur >= max_dur:
                    if local_starting_point == windowSize:
                        print('    STEP1 ',max_size,max_dur,local_starting_point)
                    else:
                        print('          ',max_size,max_dur,local_starting_point)

                    # Generate data with size 10000 with final name of 
                    # 'name_counter'
                    size_list.append(max_size)
                    seg_list.append([date,seg])
                    starting_point_list.append(local_starting_point)
                    timeSlide_list.append(timeSlides)
                    local_starting_point+=max_dur
                    utilised_dur -= max_dur
                    #Update the the values for the next set
                    test_total += max_size

                    name_list.append(name+'_No'+str(set_num)+'_'+str(max_size))
                    set_num+=1

                    if test_total >= size: break
                if test_total >= size: break
                # Step 2: Now we have some more left to utilise (less than the maximum) and we 
                # make a test file out of that.
                if utilised_dur>=timeSlides*duration:
                    if timeSlides==1: 
                        spare_size = utilised_dur//duration
                        spare_dur = utilised_dur
                    if timeSlides%2 == 0: 
                        spare_size = timeSlides*(timeSlides-2)*((utilised_dur//duration)//(timeSlides))
                        spare_dur = timeSlides*duration*(spare_size//(timeSlides*(timeSlides-2)))
                    if timeSlides%2 != 0 and timeSlides !=1 :
                        spare_size = timeSlides*(timeSlides-1)*((utilised_dur//duration)//(timeSlides))
                        spare_dur = timeSlides*duration*(spare_size//(timeSlides*(timeSlides-1)))

                    print('    STEP2 ',spare_size,spare_dur,local_starting_point)

                    size_list.append(spare_size)
                    seg_list.append([date,seg])
                    starting_point_list.append(local_starting_point)
                    timeSlide_list.append(timeSlides)
                    local_starting_point+= spare_dur

                    test_total += spare_size
                    name_list.append(name+'_No'+str(set_num)+'_'+str(spare_size))
                    set_num+=1

                    if test_total >= size: break
                                                            
                # Step 3:
                if utilised_dur_left > 2*duration:

                    _timeSlides = utilised_dur_left//duration

                    if _timeSlides==1: 
                        spare_size = utilised_dur_left//duration
                        spare_dur = utilised_dur_left
                    if _timeSlides%2 == 0: 
                        spare_size = _timeSlides*(_timeSlides-2)*((utilised_dur_left//duration)//(_timeSlides))
                        spare_dur = _timeSlides*duration*(spare_size//(_timeSlides*(_timeSlides-2)))
                    if _timeSlides%2 != 0 and _timeSlides !=1 :
                        spare_size = _timeSlides*(_timeSlides-1)*((utilised_dur_left//duration)//(_timeSlides))
                        spare_dur = _timeSlides*duration*(spare_size//(_timeSlides*(_timeSlides-1))) 
                    print('    STEP3 ',spare_size,spare_dur,local_starting_point,'timeSlide =',_timeSlides)

                    size_list.append(spare_size)
                    seg_list.append([date,seg])
                    starting_point_list.append(local_starting_point)
                    timeSlide_list.append(_timeSlides)
                    local_starting_point+= spare_dur

                    test_total += spare_size
                    name_list.append(name+'_No'+str(set_num)+'_'+str(spare_size))
                    set_num+=1
                if test_total >= size: break


            if date_counter==len(date_list): date_counter=0 
            if len(date_list) == 1: date_counter=0
            date=date_list[date_counter]


        

        d={'segment' : seg_list, 'size' : size_list 
           , 'start_point' : starting_point_list, 'timeSlides' : timeSlide_list
           , 'name' : name_list}
        
        if test_total>size:
            d['size'][-1] -= test_total-size
            d['name'][-1] = '_'.join(d['name'][-1].split('_')[:-1])+'_'+str(d['size'][-1])
    
        print('These are the details of the data to be used for the false alarm rate test: \n')
        for i in range(len(d['segment'])):
            
            diff=(d['start_point'][i] 
                  + d['size'][i]//(d['timeSlides'][i]-1)
                  -int((d['segment'][i][1].split('_')[-1][:-4])))
            
            if diff >= 0:
                print(d['segment'][i], d['size'][i], d['start_point'][i] 
                      ,d['name'][i],d['timeSlides'][i],'FAIL BY'+30*'-'+'>',diff)
            else:
                print(d['segment'][i], d['size'][i], d['start_point'][i] 
                      ,d['name'][i],d['timeSlides'][i],'OK ')
                
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
        print('Type the name of the temporary directory:')
        dir_name = '0 0'
        while not dir_name.isidentifier():
            dir_name=input()
            if not dir_name.isidentifier(): print("Not valid Folder name ...")
        
    path = savePath
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
                print('Test is cancelled\n')
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
    
    print('Creation of temporary directory complete: '+path+dir_name)
    
    # Accounting group options
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
        
    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o3.burst.grb.xoffline'
        print("Accounting group set to 'ligo.dev.o3.burst.grb.xoffline")
    
    kwstr=""
    for k in kwargs:
        kwstr+=(","+k+"="+str(kwargs[k]))   
        
    for i in range(len(d['size'])):

        with open(path+dir_name+'/test_'+d['name'][i]+'.py','w+') as f:
            f.write('#! /usr/bin/env python3\n')
            f.write('import sys \n')
            
            f.write('sys.path.append(\'/home/'+accounting_group_user+'/mly/\')\n')

            f.write('from mly.validators import *\n\n')

            f.write("import time\n\n")
            f.write("t0=time.time()\n")
            if backgroundType=='optimal':
                
                command=( "TEST = Validator.falseAlarmTest(\n"
                         +24*" "+"models = "+str(model)+"\n"
                         +24*" "+",duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         +24*" "+",size = "+str(d['size'][i])+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                         +24*" "+",windowSize ="+str(windowSize)+"\n"
                         +24*" "+",startingPoint = "+str(d['start_point'][i])+"\n"
                         +24*" "+",name = '"+str(d['name'][i])+"_"+str(d['size'][i])+"'\n"
                         +24*" "+",savePath ='"+savePath+dir_name+"/'\n"
                         +24*" "+",plugins ="+str(plugins)+"\n"
                         +24*" "+",mapping ="+str(mapping)+kwstr+")\n")
                                                

                
            else:
                #f.write("sys.path.append('"+date_list_path+"/')\n")

                command=( "TEST = Validator.falseAlarmTest(\n"
                         +24*" "+"models = "+str(model)+"\n"
                         +24*" "+",duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         +24*" "+",size = "+str(d['size'][i])+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                         +24*" "+",noiseSourceFile = ['"+date_list_path+str(d['segment'][i])[2:]+"\n"
                         +24*" "+",windowSize ="+str(windowSize)+"\n"
                         +24*" "+",timeSlides ="+str(d['timeSlides'][i])+"\n"
                         +24*" "+",startingPoint = "+str(d['start_point'][i])+"\n"
                         +24*" "+",name = '"+str(d['name'][i])+"'\n"
                         +24*" "+",savePath ='"+savePath+dir_name+"/'\n"
                         +24*" "+",plugins ="+str(plugins)+"\n"
                         +24*" "+",mapping ="+str(mapping)+kwstr+")\n")

                                                
            f.write(command+'\n\n')
            f.write("print(time.time()-t0)\n")
        
        os.system('chmod 777 '+path+dir_name+'/test_'+d['name'][i]+'.py')
        job = Job(name='partOfGeneratio_'+str(i)
               ,executable=path+dir_name+'/test_'+d['name'][i]+'.py'
               ,submit=submit
               ,error=error
               ,output=output
               ,log=log
               ,getenv=True
               ,dag=dagman
               ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])

        job_list.append(job)

    with open(path+dir_name+'/'+name+'_info.txt','w+') as f3:
        f3.write('INFO ABOUT TEST DATA \n\n')
        f3.write('fs: '+str(fs)+'\n')
        f3.write('duration: '+str(duration)+'\n')
        f3.write('window: '+str(windowSize)+'\n')
        f3.write('timeSlides: '+str(timeSlides)+'\n'+'\n')
        
        if backgroundType != 'optimal':
            f3.write('timeSlides: '+str(timeSlides)+'\n'+'\n')
            for i in range(len(d['size'])):
                f3.write(d['segment'][i][0]+' '+d['segment'][i][1]
                         +' '+str(d['size'][i])+' '
                         +str(d['start_point'][i])+'_'+d['name'][i]+'\n')
                
    with open(path+dir_name+'/finalise_test.py','w+') as f4:
        f4.write("#! /usr/bin/env python3\n")
        f4.write("import sys \n")
        
        f4.write('sys.path.append(\'/home/'+accounting_group_user+'/mly/\')\n')
        
        f4.write("from mly.validators import *\n")
        f4.write("finalise_far('"+path+dir_name+"')\n")
        
    os.system('chmod 777 '+path+dir_name+'/finalise_test.py')
    final_job = Job(name='finishing'
               ,executable=path+dir_name+'/finalise_test.py'
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


    

    
def finalise_far(path,generation=True,forceMerging=False,**kwargs):
    
    if path[-1]!='/': path=path+'/' # making sure path is right
    files=dirlist(path)             # making a list of files in that path 
    merging_flag=False              # The flag that makes the fusion to happen
    print('Running diagnostics for file: '+path+'  ... \n') 
    pyScripts=[]
    farTests=[]
    for file in files:
        if (file[-3:]=='.py') and ('test_' in file):
            pyScripts.append(file)
        if file[-4:]=='.pkl': 
            farTests.append(file)
    # Checking if all files that should have been generated 
    # from auto_test are here
    print(len(farTests),len(pyScripts))
    if len(farTests)==len(pyScripts):
        print('Files succesfully generated, all files are here')
        print(len(farTests),' out of ',len(pyScripts))
        merging_flag=True  # Declaring that merging can happen now
    
    # If some files haven't been generated it will show a failing message
    # with the processes that failed
    else:
        failed_pyScripts=[]
        print('The following scripts failed to procced:')
        for i in range(len(pyScripts)):
            pyScripts_id=pyScripts[i][:-2]
            counter=0
            for farTest in farTests:
                if pyScripts_id in farTest:
                    counter=1
            if counter==0:
                print('rm '+pyScripts[i])
                failed_pyScripts.append(pyScripts[i])
        print('Number of failed scripts: ',len(failed_pyScripts))
        
    # Accounting group options
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
        
    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o3.burst.grb.xoffline'
        print("Accounting group set to 'ligo.dev.o3.burst.grb.xoffline")
                        
    if generation==False: return
    
    if forceMerging==True:
        merging_flag=True
            
    if merging_flag==False:

        if os.path.isfile(path+'/'+'.flag_file.sh'):
            print("\nThe following scripts failed to run trough:\n")
            for failed_pyScript in failed_pyScripts:
                print(failed_pyScript+"\n") 
                
            return


        with open(path+'/'+'.flag_file.sh','w+') as f2:
             f2.write('#!/usr/bin/bash +x\n\n')

        error = path+'condor/error'
        output = path+'condor/output'
        log = path+'condor/log'
        submit = path+'condor/submit'

        repeat_dagman = Dagman(name='repeat_falsAlarmDagman',
                submit=submit)
        repeat_job_list=[]


        for i in range(len(failed_pyScripts)):

            repeat_job = Job(name='repeat_partOfGeneratio_'+str(i)
                       ,executable=path+failed_pyScripts[i]
                       ,submit=submit
                       ,error=error
                       ,output=output
                       ,log=log
                       ,getenv=True
                       ,dag=repeat_dagman
                       ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])

            repeat_job_list.append(repeat_job)
               
        repeat_final_job = Job(name='repeat_finishing'
                           ,executable=path+'finalise_test.py'
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
        totalTestSize=0
        for k in range(len(farTests)):
            if k==0:
                with open(path+farTests[k],'rb') as obj:
                    finaltest = pickle.load(obj)

            else:
                try:
                    with open(path+farTests[k],'rb') as obj:
                        part_of_test = pickle.load(obj)
                except EOFError:
                    continue
                    
                finaltest = finaltest.append(part_of_test)
                print(k)

        finaltest = finaltest.sort_values(by=['total'],ascending = False)
        with open(path+'FAR_TEST.pkl', 'wb') as output:
            pickle.dump(finaltest, output, 4)
                

        # Deleting unnescesary file in the folder
        for file in dirlist(path):
            if (('.out' in file) or ('.py' in file)
                or ('part_of' in file) or ('No' in file) or ('test' in file) or ('.sh' in file) or ('10000' in file)):
                os.system('rm '+path+file)
        
        print('Total Size is :',totalTestSize )
        print(str(totalTestSize/(365*24*3600))+' years')

    
    

    
    

    
    
def online_FAR(model
             ,duration 
             ,fs
             ,detectors
             ,size
             ,dates=None
             ,backgroundType = None
             ,windowSize = None #(32)            
             ,destinationFile = None
             ,mapping=None
             ,plugins=None
             ,externalLagSize=None
             ,maxExternalLag=None
             ,strides=None
             ,restriction=None
             ,frames=None
             ,finalDirectory=None
             ,channels=None
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
        raise ValuError("size must be a possitive integer.")
        
    # ---------------------------------------------------------------------------------------- #    
    # --- backgroundType --------------------------------------------------------------------- #

    if backgroundType == None:
        backgroundType = 'optimal'
    elif not (isinstance(backgroundType,str) 
          and (backgroundType in ['optimal','sudo_real','real'])):
        raise ValueError("backgroundType is a string that can take values : "
                        +"'optimal' | 'sudo_real' | 'real'.")          
        
    # ---------------------------------------------------------------------------------------- #    
    # --- dates ------------------------------------------------------------------------------ #
    
    if backgroundType != 'optimal':
        gps_start = to_gps(dates[0])
        gps_end = to_gps(dates[1])

        if gps_start > gps_end: raise ValueError("Not valid date.")

         
    
    # ---------------------------------------------------------------------------------------- #    
    # --- windowSize --(for PSD)-------------------------------------------------------------- #        

    if windowSize == None: windowSize = int(16*duration)
    if not isinstance(windowSize,int):
        raise ValueError('windowSize needs to be an integral')
    if windowSize < duration :
        raise ValueError('windowSize needs to be bigger than the duration')


    # ---------------------------------------------------------------------------------------- #    
    # --- destinationFile -------------------------------------------------------------------- #

    if destinationFile == None : 
        destinationFile = './'#os.getcwd()
    elif (destinationFile,str): 
        if not os.path.isdir(destinationFile) : 
            raise FileNotFoundError('No such file or directory:' +destinationFile)
    else:
        raise TypeError("Destination Path has to be a string valid path")
#     if destinationFile[:5]!='/home':
#         destinationFile = os.getcwd()+'/'+destinationFile       
    if destinationFile[-1] != '/' : destinationFile=destinationFile+'/'
        
        
    # ---------------------------------------------------------------------------------------- #    
    # --- externalLagSize -------------------------------------------------------------------- #
    
    if externalLagSize==None: externalLagSize=duration*64
    
    if externalLagSize%duration!=0 or externalLagSize/duration<=2:
        raise ValueError("externalLagSize has to be a multiple of duration and at least"
                         +" three times the duration")
    
    # ---------------------------------------------------------------------------------------- #    
    # --- maxExternalLag --------------------------------------------------------------------- #
    
    if maxExternalLag==None: maxExternalLag=3600
    # We want the max external lag without the window correction to 
    # be a multiple of the externalLagSize
    if (maxExternalLag-windowSize+duration)%externalLagSize != 0:
        maxExternalLag=maxExternalLag+externalLagSize-(
            (maxExternalLag-windowSize+duration)%externalLagSize)
            
    print('MaxExternalLag:',maxExternalLag
                           ,(maxExternalLag-windowSize+duration)%externalLagSize)
    # ---------------------------------------------------------------------------------------- #    
    # --- strides ---------------------------------------------------------------------------- #
            
    if strides==None:
        strides=1
    elif not (isinstance(strides,int) and strides <= fs and fs%strides==0):
        raise ValueError('Strides must be a divisible integer of sample frequency')
        
    # ---------------------------------------------------------------------------------------- #    
    # --- restriction ------------------------------------------------------------------------ #
            
    if restriction == None: restriction ==0
    
    if backgroundType == 'optimal':
        sizeList=[10000]*(int(size/10000)+int(size%10000!=0))
    else:

        # Feching the segment times tha detectors were active
        det_flags = {'H1': 'H1:DMT-ANALYSIS_READY:1'
                       ,'L1': 'L1:DMT-ANALYSIS_READY:1'
                       ,'V1': 'V1:ITF_SCIENCE:1'}

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

        injectionList=[cbc_inj_flags
                       ,burst_inj_flags
                       ,detchar_inj_flags
                       ,stoch_inj_flags
                       ,trans_inj_flags]


        det_segs=[]
        for d in range(len(detectors)):
            main_seg=query_segments(det_flags[detectors[d]+'1'],gps_start, gps_end)['active']
            for inj in injectionList:
                try:
                    main_seg = main_seg & ~query_segments(inj[detectors[d]+'1'], gps_start, gps_end)['acitve']
                except KeyError:
                    pass
                except:
                    raise

            det_segs.append(main_seg)

        sim_seg = det_segs[0]
        for seg in det_segs[1:]:
            sim_seg = sim_seg & seg

        # Breaking up segments that are bigger than the maxExternalLag
        new_sim_seg=[]
        for seg in sim_seg:
            segsize=seg[1]-seg[0]
            if segsize>maxExternalLag:
                breaks=int(segsize/maxExternalLag)
                for k in range(breaks):
                    new_sim_seg.append(Segment(seg[0]+k*maxExternalLag,seg[0]
                                               +(k+1)*maxExternalLag))
                # The remaining segment without window correction must also be
                # a multiple of externalLagSize
                remnant_size=seg[1]-(seg[0]+(k+1)*maxExternalLag)
                if remnant_size>=externalLagSize+windowSize-duration:
                    remnant_size=remnant_size-(remnant_size-windowSize+duration)%externalLagSize
                    new_sim_seg.append(Segment(seg[0]+(k+1)*maxExternalLag
                                              ,seg[0]+(k+1)*maxExternalLag+remnant_size))
            elif segsize>=externalLagSize+windowSize-duration:
                # These smaller segments without window correction must also be
                # a multiple of externalLagSize
                remnant_size=segsize-(segsize-windowSize+duration)%externalLagSize
                new_sim_seg.append(Segment(seg[0]
                                          ,seg[0]+remnant_size))               
                
                
        #print('NEW SIM SEG: ',list(seg[1]-seg[0] for seg in new_sim_seg))
        #print('NEW SIM SEG SIZE VERIFICATION: ',list((seg[1]-seg[0]-windowSize+duration)%externalLagSize for seg in new_sim_seg))

        print("Number of segments: ",len(new_sim_seg))#list(seg[1]-seg[0] for seg in new_sim_seg),len(new_sim_seg))
        # Calculating all the external lags for each segment and putting them all together.
        externalIndeces={}
        for det in detectors: externalIndeces[det]=[]
        
        utilised_segs=0
        for seg in new_sim_seg:
            # externalLagSize is plus the windowSize because we need the first 
            # windowSize seconds in internal lags.
            #ind=externalLags(detectors,externalLagSize+windowSize-duration,seg[1]-seg[0])
            ind=internalLags(detectors,externalLagSize
                             ,seg[1]-seg[0]-windowSize+duration,includeZeroLag=False)
            for key in ind:
                externalIndeces[key]+=(np.array(ind[key])+seg[0]).tolist()
            
            blocks=len(externalIndeces[detectors[0]])
            #print((seg[1]-seg[0]),((seg[1]-seg[0])-(windowSize-duration))%externalLagSize)
            
        # # Random indexes to shuffle the externalIndeces as a group
        # randindex=np.arange(len(externalIndeces[detectors[0]]))
        # randindex = np.random.permutation(randindex) - stoped that because I want to use at least once everything

        # # Shuffling with the new indeces
        # for det in detectors:
        #     externalIndeces[det]=np.array(externalIndeces[det])[randindex]


        maximumPossibleSize=len(externalIndeces[detectors[0]])*len(circularTimeSlides(
                    detectors,int(externalLagSize/duration)))*int(externalLagSize/duration)
        
        if size>maximumPossibleSize:
            raise ValueError("Size is bigger than max possible size")
        
        print("Target Size            : ",size)
        print("Maximum possible size  : ",str(maximumPossibleSize))#+" = "+str(int(maximumPossibleSize/3600/24))+" days")
        print("Maximum external lag   : ",maxExternalLag,"s")
        print("Number of blocks       : ",blocks)

        
        internalLagSize=int(size/(len(externalIndeces[detectors[0]])*int(externalLagSize/duration)))+1
        sizeList=[ceil(size/blocks)]*blocks
                  
        

        print("Utilised time slides   : ", internalLagSize)
        print("Current block size.    : ", sizeList[0])
        print("Maximum block size     : ", len(circularTimeSlides(detectors
                                              ,int(externalLagSize/duration)))
                                                              *externalLagSize)
                    
            
    # Accounting group options
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
        
    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o3.burst.grb.xoffline'
        print("Accounting group set to 'ligo.dev.o3.burst.grb.xoffline")
    


    answers = ['no','n', 'No','NO','N','yes','y','YES','Yes','Y','exit']

    if finalDirectory==None:
        dir_name = '0 0'
    else:
        dir_name = finalDirectory

    if not dir_name.isidentifier(): raise TypeError("Not valid Folder name ...")

    print("The current path of the directory is: \n"+destinationFile+dir_name+"\n" )  

    if os.path.isdir(destinationFile+dir_name):
        if (".." in destinationFile+dir_name
            or dir_name=="home" 
            or dir_name=="home/" 
            or dir_name==accounting_group_user
            or dir_name==accounting_group_user+"/"):
            raise TypeError("You were about to delete "
                            +destinationFile+dir_name
                            +". An error was raised for safety."
                            +" Make sure that this is not an"
                            +" important directory")
        else:
            os.system('rm -r '+destinationFile+dir_name)

    print('Initiating procedure ...')
    os.system('mkdir '+destinationFile+dir_name)
    os.system('chmod -R 777 '+destinationFile+dir_name)
    os.chdir(destinationFile+dir_name)

    os.system('mkdir condor')
    os.system('chmod 777 condor')

    error = 'condor/error'
    output = 'condor/output'
    log = 'condor/log'
    submit = 'condor/submit'

    os.system('mkdir '+error)
    os.system('mkdir '+output)
    os.system('mkdir '+log)
    os.system('mkdir '+submit)

    os.system('chmod 777 '+error)
    os.system('chmod 777 '+output)
    os.system('chmod 777 '+log)
    os.system('chmod 777 '+submit)


    dagman = Dagman(name='falsAlarmDagman',
            submit=submit)
    job_list=[]

    if backgroundType == 'optimal':
        print('Creation of temporary directory complete: '+destinationFile+dir_name)
        print('Expected jobs :',str(int(size/10000)+int(size%10000!=0)))
        #size_=10000
        #looper=range(int(size/size_)+int(size%size_!=0))
        #print('Creation of temporary directory complete: '+destinationFile+dir_name)
        #print('Expected jobs :',str(int(size/size_)+int(size%size_!=0)))
    else:
        print('Creation of temporary directory complete: '+destinationFile+dir_name)
        #print('Expected jobs :',str(len(externalIndeces[detectors[0]])))
        print('Expected jobs :',len(sizeList))
        print('Internal lags :',str(internalLagSize))
        #looper=range(len(sizeList)

    target_size=0

    kwstr=""
    for k in kwargs:
        kwstr+=(","+k+"="+str(kwargs[k]))        

    for i in range(len(sizeList)):
        #target_size+=size_
        #if target_size>size: size_=size_-(target_size-size)
        target_size+=sizeList[i]
        if target_size>size:
            sizeList[i]=sizeList[i]-(target_size-size)
        if sizeList[i] <=0: break


        with open('test_'+str(i)+'.py','w+') as f:
            f.write('#! /usr/bin/env python3\n')
            f.write('import sys \n')
            
            f.write('sys.path.append(\'/home/'+accounting_group_user+'/mly/\')\n')

            f.write('from mly.validators import *\n\n')

            f.write("import time\n\n")
            f.write("t0=time.time()\n")

            if backgroundType == 'optimal':
                command=( "TEST = Validator.falseAlarmTest(\n"
                         +24*" "+"models = "+str(model)+"\n"
                         +24*" "+",duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         #+24*" "+",size = "+str(size_)+"\n"
                         +24*" "+",size = "+str(sizeList[i])+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                         +24*" "+",windowSize ="+str(windowSize)+"\n"
                         +24*" "+",name = 'test_"+str(i)+"'\n"
                         +24*" "+",savePath ='"+destinationFile+dir_name+"/'\n"
                         +24*" "+",plugins ="+str(plugins)+"\n"
                         +24*" "+",mapping ="+str(mapping)+"\n"
                         +24*" "+",strides ="+str(strides)+"\n"
                         +24*" "+",restriction ="+str(restriction)+kwstr+")\n")
            else:
                command=( "TEST = Validator.falseAlarmTest(\n"
                         +24*" "+"models = "+str(model)+"\n"
                         +24*" "+",duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         #+24*" "+",size = "+str(size_)+"\n"
                         +24*" "+",size = "+str(sizeList[i])+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                         +24*" "+",noiseSourceFile = "+str(list([externalIndeces[det][i],externalIndeces[det][i]
                                                      +int(externalLagSize+windowSize-duration)] for det in detectors))+"\n"
                         +24*" "+",windowSize ="+str(windowSize)+"\n"
                         +24*" "+",timeSlides ="+str(internalLagSize)+"\n"
                         +24*" "+",startingPoint = "+str(0)+"\n"
                         +24*" "+",name = 'test_"+str(i)+"'\n"
                         +24*" "+",plugins ="+str(plugins)+"\n"
                         +24*" "+",mapping ="+str(mapping)+"\n"
                         +24*" "+",strides ="+str(strides)+"\n"
                         +24*" "+",restriction ="+str(restriction)+"\n"
                         +24*" "+",frames ='"+str(frames)+"'\n"
                         +24*" "+",channels ='"+str(channels)+"'"+kwstr+")\n")


            f.write(command+'\n\n')
            f.write("print(time.time()-t0)\n")

        os.system('chmod 777 test_'+str(i)+'.py')
        job = Job(name='partOfFAR_'+str(i)
               ,executable='test_'+str(i)+'.py'
               ,submit=submit
               ,error=error
               ,output=output
               ,log=log
               ,getenv=True
               ,dag=dagman
               ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])

        job_list.append(job)
            
    with open('finalise_test.py','w+') as f4:
        f4.write("#! /usr/bin/env python3\n")
        f4.write("import sys \n")
        
        f4.write('sys.path.append(\'/home/'+accounting_group_user+'/mly/\')\n')
        f4.write("from mly.validators import *\n")
        f4.write("finalise_far('.')\n")
        
    os.system('chmod 777 finalise_test.py')
    final_job = Job(name='finishing'
               ,executable='finalise_test.py'
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

    print('All set. Initiate dataset generation y/n?')
    if finalDirectory==None:
        answer4=input()
    else:
        answer4='y'

    if answer4 in ['yes','y','YES','Yes','Y']:
        print('Creating Job queue')
        print(os.getcwd())
        dagman.build_submit()

        return
    
    else:
        print('Data generation canceled')
        #os.system('rm -r '+destinationFile+dir_name)
        return


    
    
def zeroLagSearch(model
                 ,duration 
                 ,fs
                 ,detectors
                 ,dates=None
                 ,windowSize = None #(32)            
                 ,destinationFile = None
                 ,mapping=None
                 ,plugins=None
                 ,restriction=None
                 ,strides=None
                 ,destinationFileName=None
                 ,frames=None 
                 ,channels=None
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
    # --- dates ------------------------------------------------------------------------------ #
    
    gps_start = to_gps(dates[0])
    gps_end = to_gps(dates[1])
    
    if gps_start > gps_end: raise ValueError("Not valid date.")
              
    
    # ---------------------------------------------------------------------------------------- #    
    # --- windowSize --(for PSD)-------------------------------------------------------------- #        

    if windowSize == None: windowSize = int(16*duration)
    if not isinstance(windowSize,int):
        raise ValueError('windowSize needs to be an integral')
    if windowSize < duration :
        raise ValueError('windowSize needs to be bigger than the duration')


    # ---------------------------------------------------------------------------------------- #    
    # --- destinationFile -------------------------------------------------------------------- #

    if destinationFile == None : 
        destinationFile = os.getcwd()
    elif (destinationFile,str): 
        if not os.path.isdir(destinationFile) : 
            raise FileNotFoundError('No such file or directory:' +destinationFile)
    else:
        raise TypeError("Destination Path has to be a string valid path")
    if destinationFile[:5]!='/home':
        destinationFile = os.getcwd()+'/'+destinationFile       
    if destinationFile[-1] != '/' : destinationFile=destinationFile+'/'
        
    # ---------------------------------------------------------------------------------------- #    
    # --- strides ---------------------------------------------------------------------------- #
            
    if strides==None:
        strides=1
    elif not (isinstance(strides,int) and strides <= fs and fs%strides==0):
        raise ValueError('Strides must be a divisible integer of sample frequency')
        
    # ---------------------------------------------------------------------------------------- #    
    # --- restriction ------------------------------------------------------------------------ #
            
    if restriction == None: restriction ==0
        
    # Accounting group options
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
        
    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o3.burst.grb.xoffline'
        print("Accounting group set to 'ligo.dev.o3.burst.grb.xoffline")
    
    kwstr=""
    for k in kwargs:
        kwstr+=(","+k+"="+str(kwargs[k]))       

    
    det_flags = {'H1': 'H1:DMT-ANALYSIS_READY:1'
                   ,'L1': 'L1:DMT-ANALYSIS_READY:1'
                   ,'V1': 'V1:ITF_SCIENCE:1'}

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

    injectionList=[cbc_inj_flags
                   ,burst_inj_flags
                   ,detchar_inj_flags
                   ,stoch_inj_flags
                   ,trans_inj_flags]


    det_segs=[]
    for d in range(len(detectors)):
        main_seg=query_segments(det_flags[detectors[d]+'1'], gps_start,gps_end)['active']
        for inj in injectionList:
            try:
                main_seg = main_seg & ~query_segments(inj[detectors[d]+'1'], gps_start,gps_end)['active']
            except KeyError:
                 pass
            except:
                raise

        det_segs.append(main_seg)

    sim_seg = det_segs[0]
    for seg in det_segs[1:]:
        sim_seg = sim_seg & seg
    
    new_sim_seg=[]
    maxsegsize=int(16384*duration/strides)

    for seg in sim_seg:
        segsize=seg[1]-seg[0]
        print('     ',segsize,int(segsize/maxsegsize))
        
        if segsize>maxsegsize:
            breaks=int(segsize/maxsegsize)
            new_sim_seg.append(Segment(seg[0],seg[0]+maxsegsize+windowSize-duration))
            print(0,new_sim_seg[-1][1]-new_sim_seg[-1][0])

            for k in range(1,breaks):
                new_sim_seg.append(Segment(seg[0]+k*maxsegsize
                                          ,seg[0]+(k+1)*maxsegsize+windowSize-duration))
                print(1,new_sim_seg[-1][1]-new_sim_seg[-1][0])
            if breaks ==1: k =1
            if seg[1]-(seg[0]+(k+1)*maxsegsize)>windowSize:
                new_sim_seg.append(Segment(seg[0]+(k+1)*maxsegsize,seg[1]))
                print(2,new_sim_seg[-1][1]-new_sim_seg[-1][0])

        elif segsize>=windowSize+duration:
            new_sim_seg.append(seg)
            print(3,new_sim_seg[-1][1]-new_sim_seg[-1][0])



    answers = ['no','n', 'No','NO','N','yes','y','YES','Yes','Y','exit']
    
    if destinationFileName == None:
        print('Type the name of the temporary directory:')
        dir_name = '0 0'
        while not dir_name.isidentifier():
            dir_name=input()
            if not dir_name.isidentifier(): print("Not valid Folder name ...")
    else: 
        dir_name=destinationFileName
        
    print("The current path of the directory is: \n"+destinationFile+dir_name+"\n" )  
    answer = None
    while answer not in answers:
        print('Do you accept the path y/n ?')
        answer=input()
        if answer not in answers: print("Not valid answer ...")

    if answer in ['no','n', 'No','NO','N','exit']:
        print('Exiting procedure ...')
        return

    elif answer in ['yes','y','YES','Yes','Y']:
        if os.path.isdir(destinationFile+dir_name):
            answer = None
            while answer not in answers:
                print('Already existing '+dir_name+' directory, do you want to'
                      +' overwrite it? y/n')
                answer=input()
                if answer not in answers: print("Not valid answer ...")
            if answer in ['yes','y','YES','Yes','Y']:
                os.system('rm -r '+destinationFile+dir_name)
            elif answer in ['no','n', 'No','NO','N']:
                print('Test is cancelled\n')
                print('Exiting procedure ...')
                return
    
        print('Initiating procedure ...')
        os.system('mkdir '+destinationFile+dir_name)

        error = destinationFile+dir_name+'/condor/error'
        output = destinationFile+dir_name+'/condor/output'
        log = destinationFile+dir_name+'/condor/log'
        submit = destinationFile+dir_name+'/condor/submit'

        dagman = Dagman(name='falsAlarmDagman',
                submit=submit)
        job_list=[]
        
        target_size=0

        print('Creation of temporary directory complete: '+destinationFile+dir_name)
        print('Expected jobs :',str(len(new_sim_seg)))
              
        for i in range(len(new_sim_seg)):
                   
            _size=int(new_sim_seg[i][1]-new_sim_seg[i][0]-windowSize)# I removed duration when I introduced strides
            with open(destinationFile+dir_name+'/test_'+str(i)+'.py','w+') as f:
                f.write('#! /usr/bin/env python3\n')
                f.write('import sys \n')
                
                f.write('sys.path.append(\'/home/'+accounting_group_user+'/mly/\')\n')

                f.write('from mly.validators import *\n\n')

                f.write("import time\n\n")
                f.write("t0=time.time()\n")

                command=( "TEST = Validator.falseAlarmTest(\n"
                         +24*" "+"models = "+str(model)+"\n"
                         +24*" "+",duration = "+str(duration)+"\n"
                         +24*" "+",fs = "+str(fs)+"\n"
                         +24*" "+",size = "+str(_size)+"\n"
                         +24*" "+",detectors = "+str(detectors)+"\n"
                         +24*" "+",backgroundType = '"+str('real')+"'\n"
                         +24*" "+",noiseSourceFile = "+str(list([new_sim_seg[i][0],new_sim_seg[i][0]+int(new_sim_seg[i][1]-new_sim_seg[i][0])] for det in detectors))+"\n"
                         +24*" "+",windowSize ="+str(windowSize)+"\n"
                         +24*" "+",timeSlides ="+str(0)+"\n"
                         +24*" "+",startingPoint = "+str(0)+"\n"
                         +24*" "+",name = 'test_"+str(i)+"'\n"
                         +24*" "+",savePath ='"+destinationFile+dir_name+"/'\n"
                         +24*" "+",plugins ="+str(plugins)+"\n"
                         +24*" "+",mapping ="+str(mapping)+"\n"
                         +24*" "+",strides ="+str(strides)+"\n"
                         +24*" "+",restriction ="+str(restriction)+"\n"
                         +24*" "+",frames ='"+str(frames)+"'\n"
                         +24*" "+",channels ='"+str(channels)+"'\n"+kwstr+")\n")


                f.write(command+'\n\n')
                f.write("print(time.time()-t0)\n")

            os.system('chmod 777 '+destinationFile+dir_name+'/test_'+str(i)+'.py')
            job = Job(name='partOfGeneratio_'+str(i)
                   ,executable=destinationFile+dir_name+'/test_'+str(i)+'.py'
                   ,submit=submit
                   ,error=error
                   ,output=output
                   ,log=log
                   ,getenv=True
                   ,dag=dagman
                   ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])

            job_list.append(job)
            
    with open(destinationFile+dir_name+'/finalise_test.py','w+') as f4:
        f4.write("#! /usr/bin/env python3\n")

        f4.write("import sys \n")
        
        f4.write('sys.path.append(\'/home/'+accounting_group_user+'/mly/\')\n')
        
        f4.write("from mly.validators import *\n")
        f4.write("finalise_far('"+destinationFile+dir_name+"')\n")
        
    os.system('chmod 777 '+destinationFile+dir_name+'/finalise_test.py')
    final_job = Job(name='finishing'
               ,executable=destinationFile+dir_name+'/finalise_test.py'
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
    
    if destinationFileName == None:

        print('All set. Initiate dataset generation y/n?')
        answer4=input()

        if answer4 in ['yes','y','YES','Yes','Y']:
            print('Creating Job queue')

            dagman.build_submit()

            return

        else:
            print('Data generation canceled')
            os.system('cd')
            os.system('rm -r '+destinationFile+dir_name)
            return
        
    else:
        dagman.build_submit()
        return


    
    
def online_TAR(model
             ,duration 
             ,fs
             ,detectors
             ,size
             ,injection_source
             ,injectionSNR=None
             ,dates=None
             ,backgroundType = None
             ,windowSize = None #(32)            
             ,destinationFile = None
             ,mapping=None
             ,plugins=None
             ,externalLagSize=None
             ,maxExternalLag=None
             ,strides=None
             ,restriction=None
             ,frames=None
             ,finalDirectory=None
             ,channels=None
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
        raise ValuError("size must be a possitive integer.")
        
    # ---------------------------------------------------------------------------------------- #    
    # --- backgroundType --------------------------------------------------------------------- #

    if backgroundType == None:
        backgroundType = 'optimal'
    elif not (isinstance(backgroundType,str) 
          and (backgroundType in ['optimal','sudo_real','real'])):
        raise ValueError("backgroundType is a string that can take values : "
                        +"'optimal' | 'sudo_real' | 'real'.")          
        
    # ---------------------------------------------------------------------------------------- #    
    # --- dates ------------------------------------------------------------------------------ #
    
    if backgroundType != 'optimal':
        gps_start = to_gps(dates[0])
        gps_end = to_gps(dates[1])

        if gps_start > gps_end: raise ValueError("Not valid date.")

         
    
    # ---------------------------------------------------------------------------------------- #    
    # --- windowSize --(for PSD)-------------------------------------------------------------- #        

    if windowSize == None: windowSize = int(16*duration)
    if not isinstance(windowSize,int):
        raise ValueError('windowSize needs to be an integral')
    if windowSize < duration :
        raise ValueError('windowSize needs to be bigger than the duration')


    # ---------------------------------------------------------------------------------------- #    
    # --- destinationFile -------------------------------------------------------------------- #

    if destinationFile == None : 
        destinationFile = os.getcwd()
    elif (destinationFile,str): 
        if not os.path.isdir(destinationFile) : 
            raise FileNotFoundError('No such file or directory:' +destinationFile)
    else:
        raise TypeError("Destination Path has to be a string valid path")
    if destinationFile[:5]!='/home':
        destinationFile = os.getcwd()+'/'+destinationFile       
    if destinationFile[-1] != '/' : destinationFile=destinationFile+'/'
        
        
    # ---------------------------------------------------------------------------------------- #    
    # --- externalLagSize -------------------------------------------------------------------- #
    
    if externalLagSize==None: externalLagSize=duration*64
    
    if externalLagSize%duration!=0 or externalLagSize/duration<=2:
        raise ValueError("externalLagSize has to be a multiple of duration and at least"
                         +" three times the duration")
    
    # ---------------------------------------------------------------------------------------- #    
    # --- maxExternalLag --------------------------------------------------------------------- #
    
    if maxExternalLag==None: maxExternalLag=3600
    # We want the max external lag without the window correction to 
    # be a multiple of the externalLagSize
    if (maxExternalLag-windowSize+duration)%externalLagSize != 0:
        maxExternalLag=maxExternalLag+externalLagSize-(
            (maxExternalLag-windowSize+duration)%externalLagSize)
            
    print('MaxExternalLag:',maxExternalLag
                           ,(maxExternalLag-windowSize+duration)%externalLagSize)
    # ---------------------------------------------------------------------------------------- #    
    # --- strides ---------------------------------------------------------------------------- #
            
    if strides==None:
        strides=1
    elif not (isinstance(strides,int) and strides <= fs and fs%strides==0):
        raise ValueError('Strides must be a divisible integer of sample frequency')
        
    # ---------------------------------------------------------------------------------------- #    
    # --- restriction ------------------------------------------------------------------------ #
            
    if restriction == None: restriction ==0
    
    
    # ---------------------------------------------------------------------------------------- #    
    # --- injectionSNR ----------------------------------------------------------------------- #

    if isinstance(injectionSNR,list): 
        snrInList=True
        metricLen=len(injectionSNR)
    else:
        snrInList=False

    # ---------------------------------------------------------------------------------------- #    
    # --- injectionHRSS ---------------------------------------------------------------------- #

    if 'injectionHRSS' in kwargs:
        injectionHRSS = kwargs['injectionHRSS']

    else: 
        injectionHRSS=None

    if isinstance(injectionHRSS,list): 
        hrssInList=True
        metricLen=len(injectionHRSS)
    else:
        hrssInList=False
    
    if backgroundType == 'optimal':
        sizeList=[10]*(int(size/10)+int(size%10!=0))
    else:

        # Feching the segment times tha detectors were active
        det_flags = {'H1': 'H1:DMT-ANALYSIS_READY:1'
                       ,'L1': 'L1:DMT-ANALYSIS_READY:1'
                       ,'V1': 'V1:ITF_SCIENCE:1'}

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

        injectionList=[cbc_inj_flags
                       ,burst_inj_flags
                       ,detchar_inj_flags
                       ,stoch_inj_flags
                       ,trans_inj_flags]


        det_segs=[]
        for d in range(len(detectors)):
            main_seg=query_segments(det_flags[detectors[d]+'1'],gps_start, gps_end)['active']
            for inj in injectionList:
                try:
                    main_seg = main_seg & ~query_segments(inj[detectors[d]+'1'], gps_start, gps_end)['acitve']
                except KeyError:
                    pass
                except:
                    raise

            det_segs.append(main_seg)

        sim_seg = det_segs[0]
        for seg in det_segs[1:]:
            sim_seg = sim_seg & seg

        # Breaking up segments that are bigger than the maxExternalLag
        new_sim_seg=[]
        for seg in sim_seg:
            segsize=seg[1]-seg[0]
            if segsize>maxExternalLag:
                breaks=int(segsize/maxExternalLag)
                for k in range(breaks):
                    new_sim_seg.append(Segment(seg[0]+k*maxExternalLag,seg[0]
                                               +(k+1)*maxExternalLag))
                # The remaining segment without window correction must also be
                # a multiple of externalLagSize
                remnant_size=seg[1]-(seg[0]+(k+1)*maxExternalLag)
                if remnant_size>=externalLagSize+windowSize-duration:
                    remnant_size=remnant_size-(remnant_size-windowSize+duration)%externalLagSize
                    new_sim_seg.append(Segment(seg[0]+(k+1)*maxExternalLag
                                              ,seg[0]+(k+1)*maxExternalLag+remnant_size))
            elif segsize>=externalLagSize+windowSize-duration:
                # These smaller segments without window correction must also be
                # a multiple of externalLagSize
                remnant_size=segsize-(segsize-windowSize+duration)%externalLagSize
                new_sim_seg.append(Segment(seg[0]
                                          ,seg[0]+remnant_size))               
                
                
        #print('NEW SIM SEG: ',list(seg[1]-seg[0] for seg in new_sim_seg))
        #print('NEW SIM SEG SIZE VERIFICATION: ',list((seg[1]-seg[0]-windowSize+duration)%externalLagSize for seg in new_sim_seg))

        print("Number of segments: ",len(new_sim_seg))#list(seg[1]-seg[0] for seg in new_sim_seg),len(new_sim_seg))
        # Calculating all the external lags for each segment and putting them all together.
        externalIndeces={}
        for det in detectors: externalIndeces[det]=[]
        
        utilised_segs=0
        for seg in new_sim_seg:
            # externalLagSize is plus the windowSize because we need the first 
            # windowSize seconds in internal lags.
            #ind=externalLags(detectors,externalLagSize+windowSize-duration,seg[1]-seg[0])
            ind=internalLags(detectors,externalLagSize,seg[1]-seg[0]-windowSize+duration,includeZeroLag=False)
            for key in ind:
                externalIndeces[key]+=(np.array(ind[key])+seg[0]).tolist()
            
            blocks=len(externalIndeces[detectors[0]]) #maybe move that <--|
            #print((seg[1]-seg[0]),((seg[1]-seg[0])-(windowSize-duration))%externalLagSize)
            
        # # Random indexes to shuffle the externalIndeces as a group
        # randindex=np.arange(len(externalIndeces[detectors[0]]))
        # randindex = np.random.permutation(randindex) - stoped that because I want to use at least once everything

        # # Shuffling with the new indeces
        # for det in detectors:
        #     externalIndeces[det]=np.array(externalIndeces[det])[randindex]


        maximumPossibleSize=len(externalIndeces[detectors[0]])*len(circularTimeSlides(
                    detectors,int(externalLagSize/duration)))*int(externalLagSize/duration)
        
        if size>maximumPossibleSize:
            raise ValueError("Size is bigger than max possible size")
        
        print("Target Size            : ",size)
        print("Maximum possible size  : ",str(maximumPossibleSize))#+" = "+str(int(maximumPossibleSize/3600/24))+" days")
        print("Maximum external lag   : ",maxExternalLag,"s")
        print("Number of blocks       : ",blocks)

        
        internalLagSize=int(size/(len(externalIndeces[detectors[0]])*int(externalLagSize/duration)))+1
        sizeList=[ceil(size/blocks)]*blocks
        

#         if size<=len(externalIndeces[detectors[0]])*int(externalLagSize/duration):
#             utilised_blocks=ceil(size/externalLagSize)
#             internalLagSize=0
#             sizeList=[externalLagSize]*utilised_blocks
            
#         elif size>len(externalIndeces[detectors[0]])*int(externalLagSize/duration) and size <= maximumPossibleSize:
#             utilised_blocks=blocks
#             internalLagSize=int(size/(len(externalIndeces[detectors[0]])*int(externalLagSize/duration)))
#             sizeList=[externalLagSize*(internalLagSize+1)]*utilised_blocks
#                       size/utilised_blocks
#         else:
#             raise ValueError("External lag size not big enough for desire test number")
            
            
        

        print("Utilised time slides   : ", internalLagSize)
        print("Current block size.    : ", sizeList[0])
        print("Maximum block size     : "
              , len(circularTimeSlides(detectors,int(externalLagSize/duration)))*externalLagSize)
                    
            
            
#         if size<=chunks*int(externalLagSize/duration)*(int(externalLagSize/duration)-(
#             int(int(externalLagSize/duration)%2==0)+1)):

#             for det in detectors:
#                 externalIndeces[det]=externalIndeces[det][:ceil(size/(int(externalLagSize/duration)*(int(externalLagSize/duration)-(
#                     int(int(externalLagSize/duration)%2==0)+1))))]

#             internalLagSize=(int(externalLagSize/duration)-(
#                 int(int(externalLagSize/duration)%2==0)+1))

#             size_=int(externalLagSize/duration)*(int(externalLagSize/duration)-(
#                     int(int(externalLagSize/duration)%2==0)+1))



#         # If target size is even bigger it cannot be fulfiled in this 
#         # time interval and it needs bigger date difference.
#         else:
#             raise ValueError("The date interval provided cannot fullfil the target size by "
#                              +str(duration*(size-chunks*int(
#                                  externalLagSize/duration)*(int(externalLagSize/duration)-(
#                                  int(int(externalLagSize/duration)%2==0)+1))))+" seconds")
#         print("Internal lag size      : ",internalLagSize)

    # Accounting group options
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
        
    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o3.burst.grb.xoffline'
        print("Accounting group set to 'ligo.dev.o3.burst.grb.xoffline")
    


    answers = ['no','n', 'No','NO','N','yes','y','YES','Yes','Y','exit']

    print('Type the name of the temporary directory:')
    if finalDirectory==None:
        dir_name = '0 0'
    else:
        dir_name = finalDirectory

    while not dir_name.isidentifier():
        dir_name=input()
        if not dir_name.isidentifier(): print("Not valid Folder name ...")

    print("The current path of the directory is: \n"+destinationFile+dir_name+"\n" )  
    answer = None
    while answer not in answers:
        print('Do you accept the path y/n ?')
        if finalDirectory==None:
            answer=input()
        else:
            answer='y'
        if answer not in answers: print("Not valid answer ...")

    if answer in ['no','n', 'No','NO','N','exit']:
        print('Exiting procedure ...')
        return

    elif answer in ['yes','y','YES','Yes','Y']:
        if os.path.isdir(destinationFile+dir_name):
            answer = None
            while answer not in answers:
                print('Already existing '+dir_name+' directory, do you want to'
                      +' overwrite it? y/n')
                if finalDirectory==None:
                    answer=input()
                else:
                    answer='y'
                if answer not in answers: print("Not valid answer ...")
            if answer in ['yes','y','YES','Yes','Y']:
                os.system('rm -r '+destinationFile+dir_name)
            elif answer in ['no','n', 'No','NO','N']:
                print('Test is cancelled\n')
                print('Exiting procedure ...')
                return

        print('Initiating procedure ...')
        os.system('mkdir '+destinationFile+dir_name)

        error = destinationFile+dir_name+'/condor/error'
        output = destinationFile+dir_name+'/condor/output'
        log = destinationFile+dir_name+'/condor/log'
        submit = destinationFile+dir_name+'/condor/submit'

        dagman = Dagman(name='falsAlarmDagman',
                submit=submit)
        job_list=[]
        
        if backgroundType == 'optimal':
            print('Creation of temporary directory complete: '+destinationFile+dir_name)
            print('Expected jobs :',str(int(size/10000)+int(size%10000!=0)))
            #size_=10000
            #looper=range(int(size/size_)+int(size%size_!=0))
            #print('Creation of temporary directory complete: '+destinationFile+dir_name)
            #print('Expected jobs :',str(int(size/size_)+int(size%size_!=0)))
        else:
            print('Creation of temporary directory complete: '+destinationFile+dir_name)
            #print('Expected jobs :',str(len(externalIndeces[detectors[0]])))
            print('Expected jobs :',len(sizeList))
            print('Internal lags :',str(internalLagSize))
            #looper=range(len(sizeList)
        
        target_size=0

        kwstr=""
        for k in kwargs:
            kwstr+=(","+k+"="+str(kwargs[k]))        

        for i in range(len(sizeList)):
            #target_size+=size_
            #if target_size>size: size_=size_-(target_size-size)
            target_size+=sizeList[i]
            if target_size>size: sizeList[i]=sizeList[i]-(target_size-size)
            
            if sizeList[i]<=0: continue
            
            with open(destinationFile+dir_name+'/test_'+str(i)+'.py','w+') as f:
                f.write('#! /usr/bin/env python3\n')
                f.write('import sys \n')
                
                f.write('sys.path.append(\'/home/'+accounting_group_user+'/mly/\')\n')

                f.write('from mly.validators import *\n\n')

                f.write("import time\n\n")
                f.write("t0=time.time()\n")
                  
                if backgroundType == 'optimal':
                    command=( "TEST = Validator.accuracy(\n"
                             +24*" "+"models = "+str(model)+"\n"
                             +24*" "+",duration = "+str(duration)+"\n"
                             +24*" "+",injection_source = '"+str(injection_source)+"'\n"
                             +24*" "+",injectionSNR = "+str(injectionSNR)+"\n"
                             +24*" "+",fs = "+str(fs)+"\n"
                             #+24*" "+",size = "+str(size_)+"\n"
                             +24*" "+",size = "+str(sizeList[i])+"\n"
                             +24*" "+",detectors = "+str(detectors)+"\n"
                             +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                             +24*" "+",windowSize ="+str(windowSize)+"\n"
                             +24*" "+",name = 'test_"+str(i)+"'\n"
                             +24*" "+",savePath ='"+destinationFile+dir_name+"/'\n"
                             +24*" "+",plugins ="+str(plugins)+"\n"
                             +24*" "+",mapping ="+str(mapping)+"\n"
                             +24*" "+",strides ="+str(strides)+"\n"
                             +24*" "+",restriction ="+str(restriction)+kwstr+")\n")

                    
                else:
                    randomNum=np.random.randint(0,externalLagSize-sizeList[i])
                    command=( "TEST = Validator.accuracy(\n"
                             +24*" "+"models = "+str(model)+"\n"
                             +24*" "+",duration = "+str(duration)+"\n"
                             +24*" "+",injection_source = '"+str(injection_source)+"'\n"
                             +24*" "+",injectionSNR = "+str(injectionSNR)+"\n"
                             +24*" "+",fs = "+str(fs)+"\n"
                             #+24*" "+",size = "+str(size_)+"\n"
                             +24*" "+",size = "+str(sizeList[i])+"\n"
                             +24*" "+",detectors = "+str(detectors)+"\n"
                             +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                             +24*" "+",noiseSourceFile = "+str(list([externalIndeces[det][i]+randomNum
                                                                     ,externalIndeces[det][i]+randomNum+sizeList[i]
                                                                     +int(windowSize-duration)] for det in detectors))+"\n"
                             +24*" "+",windowSize ="+str(windowSize)+"\n"
                             +24*" "+",timeSlides ="+str(internalLagSize)+"\n"
                             +24*" "+",startingPoint = "+str(0)+"\n"
                             +24*" "+",name = 'test_"+str(i)+"'\n"
                             +24*" "+",savePath ='"+destinationFile+dir_name+"/'\n"
                             +24*" "+",plugins ="+str(plugins)+"\n"
                             +24*" "+",mapping ="+str(mapping)+"\n"
                             +24*" "+",strides ="+str(strides)+"\n"
                             +24*" "+",restriction ="+str(restriction)+"\n"
                             +24*" "+",frames ='"+str(frames)+"'\n"
                             +24*" "+",channels ='"+str(channels)+"'"+kwstr+")\n")


                f.write(command+'\n\n')
                f.write("print(time.time()-t0)\n")

            os.system('chmod 777 '+destinationFile+dir_name+'/test_'+str(i)+'.py')
            job = Job(name='partOfGeneratio_'+str(i)
                   ,executable=destinationFile+dir_name+'/test_'+str(i)+'.py'
                   ,submit=submit
                   ,error=error
                   ,output=output
                   ,log=log
                   ,getenv=True
                   ,dag=dagman
                   ,retry=10
                   ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])

            job_list.append(job)
            
    with open(destinationFile+dir_name+'/finalise_test.py','w+') as f4:
        f4.write("#! /usr/bin/env python3\n")
        f4.write("import sys \n")

        f4.write('sys.path.append(\'/home/'+accounting_group_user+'/mly/\')\n')
        
        f4.write("from mly.validators import *\n")
        f4.write("finalise_tar('"+destinationFile+dir_name+"')\n")
        
    os.system('chmod 777 '+destinationFile+dir_name+'/finalise_test.py')
    final_job = Job(name='finishing'
               ,executable=destinationFile+dir_name+'/finalise_test.py'
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

    print('All set. Initiate dataset generation y/n?')
    if finalDirectory==None:
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
        os.system('rm -r '+destinationFile+dir_name)
        return

    
    
    
    
def finalise_tar(path,generation=True,forceMerging=False,**kwargs):
    
    if path[-1]!='/': path=path+'/' # making sure path is right
    files=dirlist(path)             # making a list of files in that path 
    merging_flag=False              # The flag that makes the fusion to happen
    print('Running diagnostics for file: '+path+'  ... \n') 
    pyScripts=[]
    tarTests=[]
    for file in files:
        if (file[-3:]=='.py') and ('test_' in file):
            pyScripts.append(file)
        if file[-4:]=='.pkl': 
            tarTests.append(file)
    # Checking if all files that should have been generated 
    # from auto_test are here
    print(len(tarTests),len(pyScripts))
    if len(tarTests)==len(pyScripts):
        print('Files succesfully generated, all files are here')
        print(len(tarTests),' out of ',len(pyScripts))
        merging_flag=True  # Declaring that merging can happen now
    
    # If some files haven't been generated it will show a failing message
    # with the processes that failed
    else:
        failed_pyScripts=[]
        print('The following scripts failed to procced:')
        for i in range(len(pyScripts)):
            pyScripts_id=pyScripts[i][:-2]
            counter=0
            for tarTest in tarTests:
                if pyScripts_id in tarTest:
                    counter=1
            if counter==0:
                print('rm '+pyScripts[i])
                failed_pyScripts.append(pyScripts[i])
        print('Number of failed scripts: ',len(failed_pyScripts))
        
    # Accounting group options
    if 'accounting_group_user' in kwargs:
        accounting_group_user=kwargs['accounting_group_user']
    else:
        accounting_group_user=os.environ['LOGNAME']
        
    if 'accounting_group' in kwargs:
        accounting_group=kwargs['accounting_group']
    else:
        accounting_group='ligo.dev.o3.burst.grb.xoffline'
        print("Accounting group set to 'ligo.dev.o3.burst.grb.xoffline")
                        
    if generation==False: return
    
    if forceMerging==True:
        merging_flag=True
            
    if merging_flag==False:

        if os.path.isfile(path+'/'+'.flag_file.sh'):
            print("\nThe following scripts failed to run trough:\n")
            for failed_pyScript in failed_pyScripts:
                print(failed_pyScript+"\n") 
                
            return


        with open(path+'/'+'.flag_file.sh','w+') as f2:
             f2.write('#!/usr/bin/bash +x\n\n')

        error = path+'condor/error'
        output = path+'condor/output'
        log = path+'condor/log'
        submit = path+'condor/submit'

        repeat_dagman = Dagman(name='repeat_falsAlarmDagman',
                submit=submit)
        repeat_job_list=[]


        for i in range(len(failed_pyScripts)):

            repeat_job = Job(name='repeat_partOfGeneratio_'+str(i)
                       ,executable=path+failed_pyScripts[i]
                       ,submit=submit
                       ,error=error
                       ,output=output
                       ,log=log
                       ,getenv=True
                       ,dag=repeat_dagman
                       ,extra_lines=["accounting_group_user="+accounting_group_user
                             ,"accounting_group="+accounting_group
                             ,"request_disk            = 64M"])

            repeat_job_list.append(repeat_job)
               
        repeat_final_job = Job(name='repeat_finishing'
                           ,executable=path+'finalise_test.py'
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
        totalTestSize=0
        for k in range(len(tarTests)):
            if k==0:
                with open(path+tarTests[k],'rb') as obj:
                    finaltest = pickle.load(obj)

            else:
                try:
                    with open(path+tarTests[k],'rb') as obj:
                        part_of_test = pickle.load(obj)
                except EOFError:
                    print("EOFError")
                    continue
                    
                if 'snrs' in finaltest.keys():
                    for key in list(finaltest.keys())[1:]:
                        for h in range(len(finaltest['snrs'])):
                            finaltest[key][h]+=part_of_test[key][h]
                if 'hrss' in finaltest.keys():
                    for key in list(finaltest.keys())[1:]:
                        for h in range(len(finaltest['hrss'])):
                            finaltest[key][h]+=part_of_test[key][h]

                print(k)

        with open(path+'TAR_TEST.pkl', 'wb') as output:
            pickle.dump(finaltest, output, 4)
                

        # Deleting unnescesary file in the folder
        for file in dirlist(path):
            if (('.out' in file) or ('.py' in file)
                or ('part_of' in file) or ('No' in file) or ('test' in file) or ('.sh' in file) or ('10000' in file)):
                os.system('rm '+path+file)
        

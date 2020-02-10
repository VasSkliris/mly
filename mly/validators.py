import pandas as pd
import numpy as np
import pickle
import os
import sys
import time
import random
import copy
from math import ceil

from .simulateddetectornoise import * 
from .tools import dirlist, index_combinations
from .datatools import DataPod, DataSet

from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from matplotlib.mlab import psd

from keras.models import load_model, Sequential, Model


class Validator:
     
    

    def accuracy(model
                ,duration
                ,fs
                ,size
                ,detectors
                ,injectionFolder = None
                ,labels = None
                ,backgroundType = None
                ,injectionSNR = None
                ,noiseSourceFile = None  
                ,windowSize = None #(32)            
                ,timeSlides = None #(1)
                ,startingPoint= None #(32)
                ,name = None
                ,column = 1 # Which label accuracy. Binary case is the second column for signals.
                ,savePath = None):
                
        from keras.models import load_model, Sequential, Model
        
        fl, fm=20, int(fs/2)#
      
        profile = {'H' :'aligo','L':'aligo','V':'avirgo','K':'KAGRA_Early','I':'aligo'}
        
        # ---------------------------------------------------------------------------------------- #    
        # --- model ------------------------------------------------------------------------------ #
        
        # check model here
        if isinstance(model,str) and os.path.isfile(model):    
            trained_model = load_model(model)
        else:
            trained_model = model 

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

        # ---------------------------------------------------------------------------------------- #    
        # --- size ------------------------------------------------------------------------------- #        

        if not (isinstance(size, int) and size > 0):
            raise ValuError("size must be a possitive integer.")

        # ---------------------------------------------------------------------------------------- #    
        # --- injectionFolder -------------------------------------------------------------------- #

        if injectionFolder == None:
            pass
        elif isinstance(injectionFolder, str):            
            if (('/' in injectionFolder) and os.path.isdir(injectionFolder)):
                injectionFolder_set = injectionFolder.split('/')[-1]
            elif (('/' not in injectionFolder) and any('injections' in p for p in sys.path)):
                    for p in sys.path:
                        if ('injections' in p):
                            injectionFolder_path = (p.split('injections')[0]+'injections'
                                +'/'+injectionFolder)
                            if os.path.isdir(injectionFolder_path):
                                injectionFolder = injectionFolder_path
                                injectionFolder_set = injectionFolder.split('/')[-1]
                            else:
                                raise FileNotFoundError('No such file or directory:'
                                                        +injectionFolder_path)

            else:
                raise FileNotFoundError('No such file or directory:'+injectionFolder) 
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
        if injectionFolder == None :
            raise ValueError("You need to specify source file for the injections")
        elif (injectionFolder != None and injectionSNR == None ):
            raise ValueError("If you want to use an injection for generation of"+
                             "data, you have to specify the SNR you want.")
        elif injectionFolder != None and (not isinstance(injectionSNR,(tuple,list)) 
                             and (not all(isinstance(snr,(int,float)) for snr in injectionSNR)) 
                             and injectionSNR >= 0):
            raise ValueError("injectionSNR has to be a positive number")


        # ---------------------------------------------------------------------------------------- #    
        # --- noiseSourceFile -------------------------------------------------------------------- #

        if (backgroundType == 'sudo_real' or backgroundType =='real'):

            if noiseSourceFile == None:
                raise TypeError('If you use sudo_real or real noise you need'
                    +' a real noise file as a source.')

            if (noiseSourceFile!=None and isinstance(noiseSourceFile,list) 
                    and len(noiseSourceFile)==2 
                    and all(isinstance(el,str) for el in noiseSourceFile)):

                if '.txt' in noiseSourceFile[1]:
                    noiseSourceFile[1] = noiseSourceFile[1][:-4]

                path_main=''
                path_check = False

                if (('/' in noiseSourceFile[0]) and all(os.path.isfile( noiseSourceFile[0]
                                +'/'+det+'/'+noiseSourceFile[1]+'.txt') for det in detectors)):
                    path_main = ''
                    path_check = True


                elif (('/' not in noiseSourceFile[0]) and any('ligo_data' in p for p in sys.path)):
                    for p in sys.path:
                        if ('ligo_data' in p):
                            path_main = (p.split('ligo_data')[0]+'ligo_data/'+str(int(fs))+'/')
                            if all(os.path.isfile(path_main+noiseSourceFile[0]+'/'+det+'/'
                                    +noiseSourceFile[1]+'.txt') for det in detectors):    
                                path_check = True
                                break
                            else:
                                raise FileNotFoundError("No such file or directory: "+path_main
                                                        +"/<detector>/"+noiseSourceFile[1]+".txt")
                if path_check == False:
                    raise FileNotFoundError(
                        "No such file or directory: "+noiseSourceFile[0]
                        +"/<detector>/"+noiseSourceFile[1]+".txt")
            else:
                raise TypeError("Noise source file has to be a list of two strings:\n"
                                +"--> The first is the path to the date folder that include\n "
                                +"    the data of all the detectors or just the datefile given\n"
                                +"    that the path is in sys.path.\n\n"
                                +"--> The second is the file name of the segment to be used.")
        # ---------------------------------------------------------------------------------------- #    
        # --- windowSize --(for PSD)-------------------------------------------------------------- #        

        if windowSize == None: windowSize = 32
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
            raise ValueError('lags has to be an integer')        

        # ---------------------------------------------------------------------------------------- #    
        # --- name ------------------------------------------------------------------------------- #

        if name == None : name = ''
        if not isinstance(name,str): 
            raise ValueError('name optional value has to be a string')

        # ---------------------------------------------------------------------------------------- #    
        # --- savePath -------------------------------------------------------------------- #

        if savePath == None : 
            pass
        elif (savePath,str): 
            if not os.path.isdir(savePath) : 
                raise FileNotFoundError('No such file or directory:' +savePath)
        else:
            raise TypeError("Destination Path has to be a string valid path")


            
            
            
        # Making a list of the injection names,
        # so that we can sample randomly from them

        injectionFileDict={}
        noise_segDict={}
        for det in detectors:

            if injectionFolder == None:
                injectionFileDict[det] = None
            else:
                injectionFileDict[det] = dirlist(injectionFolder+'/' + det)

        if backgroundType == 'optimal':
            magic={2048: 2**(-23./16.), 4096: 2**(-25./16.), 8192: 2**(-27./16.)}
            param = magic[fs]

        elif backgroundType in ['sudo_real','real']:
            param = 1
            for det in detectors:
                noise_segDict[det] = np.loadtxt(path_main+noiseSourceFile[0]
                                                       +'/'+det+'/'+noiseSourceFile[1]+'.txt')    
            ind=index_combinations(detectors = detectors
                               ,lags = timeSlides
                               ,length = duration
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=startingPoint)
        
                    
        snrs=[]
        acc=[]
        error=[]
        score_f=[]    
        resultDict={}
        for snr in injectionSNR: 
            DATA=DataSet()

            for I in range(size):

                detKeys = list(injectionFileDict.keys())

                if injectionFolder != None:
                    inj_ind = np.random.randint(0,len(injectionFileDict[detKeys[0]]))

                SNR0_dict={}
                back_dict={}
                inj_fft_0_dict={}
                asd_dict={}
                PSD_dict={}
                for det in detKeys:

                    if backgroundType == 'optimal':

                        # Creation of the artificial noise.
                        PSD,X,T=simulateddetectornoise(profile[det],windowSize,fs,10,fs/2)
                        PSD_dict[det]=PSD
                        back_dict[det] = X
                        # Making the noise a TimeSeries
                        back=TimeSeries(X,sample_rate=fs) 
                        # Calculating the ASD so tha we can use it for whitening later
                        asd=back.asd(1,0.5)
                        asd_dict[det] = asd


                    elif backgroundType == 'sudo_real':

                        noise_seg=noise_segDict[det]
                        # Calling the real noise segments
                        noise=noise_seg[ind[det][I]:ind[det][I]+windowSize*fs]  
                        # Generating the PSD of it
                        p, f = psd(noise, Fs=fs, NFFT=fs) 
                        p, f=p[1::],f[1::]
                        # Feeding the PSD to generate the sudo-real noise.            
                        PSD,X,T=simulateddetectornoise([f,p],windowSize,fs,10,fs/2)
                        PSD_dict[det]=PSD
                        # Making the noise a TimeSeries
                        back=TimeSeries(X,sample_rate=fs)
                        # Calculating the ASD so tha we can use it for whitening later
                        asd=back.asd(1,0.5)                 
                        asd_dict[det] = asd
                        back_dict[det] = back.value
                    elif backgroundType == 'real':

                        noise_seg=noise_segDict[det]
                        # Calling the real noise segments
                        noise=noise_seg[ind[det][I]:ind[det][I]+windowSize*fs] 
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
                        asd=back.asd(1,0.5)
                        asd_dict[det] = asd

                    #If this dataset includes injections:            
                    if injectionFolder != None:
                        # Calling the templates generated with PyCBC
                        # OLD inj=load_inj(injectionFolder,injectionFileDict[det][inj_ind], det) 
                        inj = np.loadtxt(injectionFolder+'/'+det+'/'+injectionFileDict[det][inj_ind])
                        # Saving the length of the injection
                        inj_len=len(inj)/fs                
                        # I put a random offset for all injection so that
                        # the signal is not always in the same place
                        if inj_len > duration: inj = inj[int(inj_len-duration)*fs:]
                        if inj_len < duration: inj = np.hstack((np.zeros(int(fs*(duration-inj_len)/2))
                                                            , inj
                                                            , np.zeros(int(fs*(duration-inj_len)/2))))


                        if detKeys.index(det) == 0:
                            disp = np.random.randint(-int(fs*(duration-inj_len)/2) ,int(inj_len*fs/2))                      
                        if disp >= 0: 
                            inj = np.hstack((np.zeros(int(fs*(windowSize-duration)/2)),inj[disp:]
                                                 ,np.zeros(int(fs*(windowSize-duration)/2)+disp)))   
                        if disp < 0: 
                            inj = np.hstack((np.zeros(int(fs*(windowSize-duration)/2)-disp),inj[:disp]
                                                 ,np.zeros(int(fs*(windowSize-duration)/2)))) 



                        # Calculating the one sided fft of the template,                
                        inj_fft_0=np.fft.fft(inj)
                        inj_fft_0_dict[det] = inj_fft_0
                        # we get rid of the DC value and everything above fs/2.
                        inj_fft_0N=np.abs(inj_fft_0[1:int(windowSize*fs/2)+1]) 

                        SNR0_dict[det]=np.sqrt(param*2*(1/windowSize)*np.sum(np.abs(inj_fft_0N
                                    *inj_fft_0N.conjugate())[windowSize*fl-1:windowSize*fm-1]
                                    /PSD_dict[det][windowSize*fl-1:windowSize*fm-1]))

                    else:
                        SNR0_dict[det] = 0.1 # avoiding future division with zero
                        inj_fft_0_dict[det] = np.zeros(windowSize*fs)


                # Calculation of combined SNR    
                SNR0=np.sqrt(np.sum(np.asarray(list(SNR0_dict.values()))**2))

                # Tuning injection amplitude to the SNR wanted
                podstrain = []
                for det in detectors:

                    fft_cal=(snr/SNR0)*inj_fft_0_dict[det]         
                    inj_cal=np.real(np.fft.ifft(fft_cal*fs))
                    strain=TimeSeries(back_dict[det]+inj_cal,sample_rate=fs,t0=0)
                    #Whitening final data
                    podstrain.append(((strain.whiten(1,0.5,asd=asd_dict[det])[int(((windowSize
                            -duration)/2)*fs):int(((windowSize+duration)/2)*fs)]).value).tolist())

                #podLabels={'snr': round(np.sqrt(np.sum(np.asarray(list(SNR_new.values()))**2)))}
                #labels.update(podLabels)

                DATA.add(DataPod(strain = podstrain
                                   ,fs = fs
                                   ,labels =  labels
                                   ,detectors = detKeys
                                   ,duration = duration))   
            random.shuffle(DATA.dataPods)
            
            data = DATA.unloadData(shape = (len(DATA),*model.input_shape[1:] ))
            score= trained_model.predict(data, batch_size=1)[:,column]
            resultDict[str(snr)]=score.tolist()

            snrs.append(snr)
            error.append(np.std(resultDict[str(snr)]))
            acc.append(np.int(100*np.mean(resultDict[str(snr)])))#float(np.around(np.mean(resultDict[str(snr)])*100,1)))
            score_f.append(resultDict[str(snr)])


        snrs=np.array(snrs)
        acc=np.array(acc)
        error=np.array(error)
        score_f=np.array(score_f)

        result={'snrs':snrs, 'acc': acc, 'error' :error, 'scores':score_f}
        
        
        if savePath != None:
            if savePath[-1] != '/':
                savePath = savePath+'/'
            with open(savePath+name+'.pkl', 'wb') as output:
                pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)
                
        return(result)
    
    
    
    
    
    
    
    
    def falseAlarmTest(model
                       ,duration
                       ,fs
                       ,size
                       ,detectors
                       ,backgroundType = None
                       ,noiseSourceFile = None  
                       ,windowSize = None #(32)            
                       ,timeSlides = None #(1)
                       ,startingPoint= None #(32)
                       ,name = None
                       ,savePath = None):       


       
        # Integration limits for the calculation of analytical SNR
        # These values are very important for the calculation

        fl, fm=20, int(fs/2)#
      
        profile = {'H' :'aligo','L':'aligo','V':'avirgo','K':'KAGRA_Early','I':'aligo'}
        
        # ---------------------------------------------------------------------------------------- #   
        # --- model ------------------------------------------------------------------------------ #
        
        # check model here
        if isinstance(model,str):
            if os.path.isfile(model):    
                trained_model = load_model(model)
            else:
                raise FileNotFoundError("Model file "+model+" was not found.")
        else:
            trained_model = model 

            
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
        else:
            raise ValueError("detectors have to be a list of strings or a string"+
                " with at least one the followings as elements: \n"+
                "'H' for LIGO Hanford \n'L' for LIGO Livingston\n"+
                "'V' for Virgo \n'K' for KAGRA \n'I' for LIGO India (INDIGO) \n"+
                "\n'U' if you don't want to specify detector")

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
        # --- noiseSourceFile -------------------------------------------------------------------- #

        if (backgroundType == 'sudo_real' or backgroundType =='real'):

            if noiseSourceFile == None:
                raise TypeError('If you use sudo_real or real noise you need'
                    +' a real noise file as a source.')

            if (noiseSourceFile!=None and isinstance(noiseSourceFile,list) 
                    and len(noiseSourceFile)==2 
                    and all(isinstance(el,str) for el in noiseSourceFile)):

                if '.txt' in noiseSourceFile[1]:
                    noiseSourceFile[1] = noiseSourceFile[1][:-4]

                path_main=''
                path_check = False

                if (('/' in noiseSourceFile[0]) and all(os.path.isfile( noiseSourceFile[0]
                                +'/'+det+'/'+noiseSourceFile[1]+'.txt') for det in detectors)):
                    path_main = ''
                    path_check = True


                elif (('/' not in noiseSourceFile[0]) and any('ligo_data' in p for p in sys.path)):
                    for p in sys.path:
                        if ('ligo_data' in p):
                            path_main = (p.split('ligo_data')[0]+'ligo_data/'+str(int(fs))+'/')
                            if all(os.path.isfile(path_main+noiseSourceFile[0]+'/'+det+'/'
                                    +noiseSourceFile[1]+'.txt') for det in detectors):    
                                path_check = True
                                break
                            else:
                                raise FileNotFoundError("No such file or directory: "+path_main
                                                        +"/<detector>/"+noiseSourceFile[1]+".txt")
                if path_check == False:
                    raise FileNotFoundError(
                        "No such file or directory: "+noiseSourceFile[0]
                        +"/<detector>/"+noiseSourceFile[1]+".txt")
            else:
                raise TypeError("Noise source file has to be a list of two strings:\n"
                                +"--> The first is the path to the date folder that include\n "
                                +"    the data of all the detectors or just the datefile given\n"
                                +"    that the path is in sys.path.\n\n"
                                +"--> The second is the file name of the segment to be used.")
        # ---------------------------------------------------------------------------------------- #   
        # --- windowSize --(for PSD)-------------------------------------------------------------- #        

        if windowSize == None: windowSize = 32
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
            raise ValueError('lags has to be an integer')        

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
        if savePath[-1] != '/':
                savePath = savePath+'/'
                
        # ---------------------------------------------------------------------------------------- #   


        noise_segDict={}

        if backgroundType == 'optimal':
            magic={2048: 2**(-23./16.), 4096: 2**(-25./16.), 8192: 2**(-27./16.)}
            param = magic[fs]

        elif backgroundType in ['sudo_real','real']:
            param = 1
            for det in detectors:
                noise_segDict[det] = np.loadtxt(path_main+noiseSourceFile[0]
                                                       +'/'+det+'/'+noiseSourceFile[1]+'.txt')    
            ind=index_combinations(detectors = detectors
                               ,lags = timeSlides
                               ,length = duration
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=startingPoint)

            gps0 = int(noiseSourceFile[1].split('_')[1])
        
        print(list(ind['H']))
            
        result_list=[]
        for I in range(size):
        
            back_dict={}
            asd_dict={}
            gps_dict = {}
            for det in list(detectors):

                if backgroundType == 'optimal':

                    # Creation of the artificial noise.
                    PSD,X,T=simulateddetectornoise(profile[det],windowSize,fs,10,fs/2)
                    back_dict[det] = X
                    # Making the noise a TimeSeries
                    back=TimeSeries(X,sample_rate=fs) 
                    # Calculating the ASD so tha we can use it for whitening later
                    asd=back.asd(1,0.5)
                    asd_dict[det] = asd
                    gps_dict[det] = 0.0


                elif backgroundType == 'sudo_real':

                    noise_seg=noise_segDict[det]
                    # Calling the real noise segments
                    noise=noise_seg[ind[det][I]:ind[det][I]+windowSize*fs]  
                    # Generating the PSD of it
                    p, f = psd(noise, Fs=fs, NFFT=fs) 
                    p, f=p[1::],f[1::]
                    # Feeding the PSD to generate the sudo-real noise.            
                    PSD,X,T=simulateddetectornoise([f,p],windowSize,fs,10,fs/2)
                    # Making the noise a TimeSeries
                    back=TimeSeries(X,sample_rate=fs)
                    # Calculating the ASD so tha we can use it for whitening later
                    asd=back.asd(1,0.5)                 
                    asd_dict[det] = asd
                    gps_dict[det] = gps0+ind[det][I]/fs
                    back_dict[det] = back.value
                elif backgroundType == 'real':

                    noise_seg=noise_segDict[det]
                    # Calling the real noise segments
                    noise=noise_seg[ind[det][I]:ind[det][I]+windowSize*fs] 
                    # Calculatint the psd of FFT=1s
                    p, f = psd(noise, Fs=fs,NFFT=fs)
                    # Interpolate so that has t*fs values
                    psd_int=interp1d(f,p)                                     
                    PSD=psd_int(np.arange(0,fs/2,1/windowSize))
                    # Making the noise a TimeSeries
                    back=TimeSeries(noise,sample_rate=fs)
                    back_dict[det] = back
                    # Calculating the ASD so tha we can use it for whitening later
                    asd=back.asd(1,0.5)
                    asd_dict[det] = asd
                    gps_dict[det] = gps0+ind[det][I]/fs

                
            # Tuning injection amplitude to the SNR wanted
            podstrain = []
            for det in detectors:

                strain=TimeSeries(back_dict[det] ,sample_rate=fs,t0=0)
                #Whitening final data
                podstrain.append(((strain.whiten(1,0.5,asd=asd_dict[det])[int(((windowSize
                        -duration)/2)*fs):int(((windowSize+duration)/2)*fs)]).value).tolist())
            print(len(podstrain[0]),len(podstrain[1]),len(podstrain[2]))

            podstrain= np.transpose(np.array([podstrain]),(0,2,1))
            score = trained_model.predict(podstrain, batch_size=1)[0][1]
            result_list.append([score]+list(gps_dict[det] for det in detectors))
            
            print("Test number "+str(I)+" finished with score: "+str(score))
            
        result_pd = pd.DataFrame(np.array(result_list),columns = ['SCORE']+list('GPS'+det for det in detectors))
        result_pd = result_pd.sort_values(by=['SCORE'],ascending = False)
        
        with open(savePath+name+'.pkl', 'wb') as output:
            pickle.dump(result_pd, output, pickle.HIGHEST_PROTOCOL)

        return result_pd


    
    
    
    

def auto_FAR(model
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
             ,savePath = None):

    
    # ---------------------------------------------------------------------------------------- #    
    # --- model ------------------------------------------------------------------------------ #
        
    if isinstance(model,str) and os.path.isfile(model):
        if model[:5]!='/home':
            cwd = os.getcwd()
            model = cwd+'/'+model
        trained_model = load_model(model)
    else:
        raise FileNotFoundError("Model file "+model+" was not found.")
        

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
    if size > 10000 and size%10000!=0:
        raise ValueError("For sizes above 10000 use only multiples of 10000")
    if size > 10000:
        multiples = size//10000
        size=10000
    else:
        multiples = 1
    subset_list = list('No'+str(i) for i in range(multiples))
    num_of_sets = multiples
        

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
            sys.path.append(firstDay.split(str(fs))[0]+'/'+str(fs)+'/')
            date_list_path=firstDay.split(str(fs))[0]+str(fs)
            firstDay = firstDay.split('/')[-1]
        elif '/' not in firstDay and os.path.isdir(firstDay):
            pass
        else:
            raise FileNotFoundError("No such file or directory:"+firstDay)
    else:
        raise TypeError("Path must be a string")
            
    
    # ---------------------------------------------------------------------------------------- #    
    # --- windowSize --(for PSD)-------------------------------------------------------------- #        

    if windowSize == None: windowSize = 32
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

    if name == None : name = 'untitledFalseAlarmTest'
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')

    # ---------------------------------------------------------------------------------------- #    
    # --- savePath -------------------------------------------------------------------- #

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

        if counter==len(date_list): counter=0 
        if len(date_list) == 1: counter=0
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
            local_size=ceil((duration_[i]-3*windowSize-tail_crop)/duration)
        if timeSlides%2 == 0:
            local_size=ceil((duration_[i]-3*windowSize-tail_crop)
                            /duration/timeSlides)*timeSlides*(timeSlides-2)
        if timeSlides%2 != 0 and timeSlides !=1 :
            local_size=ceil((duration_[i]-3*windowSize-tail_crop)
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
            print(set_num)
            if number_of_set_counter == size:
                number_of_set.append(subset_list[set_num])
                if size_list[-1]==size: 
                    print(set_num)
                    name_list.append(name+'_'+str(subset_list[set_num]))
                else:
                    name_list.append('part_of_'
                        +name+'_'+str(subset_list[set_num]))
                set_num+=1
                number_of_set_counter=0
                if set_num >= num_of_sets: break

            elif number_of_set_counter < size:
                number_of_set.append(subset_list[set_num])
                name_list.append('part_of_'+name+'_'+str(subset_list[set_num]))

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
                    number_of_set.append(subset_list[set_num])
                    if size_list[-1]==size: 
                        name_list.append(name+'_'+str(subset_list[set_num]))
                    else:
                        name_list.append('part_of_'
                                         +name+'_'+str(subset_list[set_num]))
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
                number_of_set.append(subset_list[set_num])
                if size_list[-1]==size: 
                    name_list.append(name+'_'+str(subset_list[set_num]))
                else:
                    name_list.append('part_of_'
                                     +name+'_'+str(subset_list[set_num]))
                set_num+=1
                if set_num >= num_of_sets: break
                number_of_set_counter=0

            elif number_of_set_counter < size:
                number_of_set.append(subset_list[set_num])
                name_list.append('part_of_'+name+'_'+str(subset_list[set_num]))


    d={'segment' : seg_list_2, 'size' : size_list 
       , 'start_point' : starting_point_list, 'set' : number_of_set
       , 'name' : name_list}

    print('These are the details of the data to be used for the false alarm rate test: \n')
    print(d['segment'][-1], d['size'][-1], d['start_point'][-1] ,d['name'][-1])
    print(len((d['segment'])), len((d['size'])),len((d['start_point'])) ,len((d['name'])))
    for i in range(len(d['segment'])):
        print(d['segment'][i], d['size'][i], d['start_point'][i] ,d['name'][i])
        
        
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
    print('Creation of temporary directory complete: '+path+dir_name)
    
    for i in range(len(d['segment'])):

        with open(path+dir_name+'/test_'+d['name'][i]+'_'
            +str(d['size'][i])+'.py','w') as f:
            f.write('#! /usr/bin/env python3\n')
            f.write('import sys \n')
            #This path is used only for me to test it
            pwd=os.getcwd()
            if 'vasileios.skliris' in pwd:
                f.write('sys.path.append(\'/home/vasileios.skliris/mly/\')\n')
                f.write("sys.path.append('"+date_list_path+"')\n")

            f.write('from mly.datatools.validators import *\n\n')

            if isinstance(d['set'][i],(float,int)):
                token_snr = str(d['set'][i])
            else:
                token_snr = '0'
            f.write("import time\n\n")
            f.write("t0=time.time()\n")

            command=( "TEST = Validator.falseAlarmTest(\n"
                     +24*" "+"model = '"+str(model)+"'\n"
                     +24*" "+",duration = "+str(duration)+"\n"
                     +24*" "+",fs = "+str(fs)+"\n"
                     +24*" "+",size = "+str(d['size'][i])+"\n"
                     +24*" "+",detectors = "+str(detectors)+"\n"
                     +24*" "+",backgroundType = '"+str(backgroundType)+"'\n"
                     +24*" "+",noiseSourceFile = "+str(d['segment'][i])+"\n"
                     +24*" "+",windowSize ="+str(windowSize)+"\n"
                     +24*" "+",timeSlides ="+str(timeSlides)+"\n"
                     +24*" "+",startingPoint = "+str(d['start_point'][i])+"\n"
                     +24*" "+",name = '"+str(d['name'][i])+"_"+str(d['size'][i])+"'\n"
                     +24*" "+",savePath ='"+savePath+dir_name+"/')\n")
            
            f.write(command+'\n\n')
            f.write("print(time.time()-t0)\n")
            f.write("sys.stdout.flush()")
    with open(path+dir_name+'/auto_far.sh','w') as f2:

        f2.write('#!/usr/bin/bash \n\n')
        f2.write("commands=()\n\n")
        
        for i in range(len(d['segment'])):

            f2.write("commands+=('"+'nohup python '+path+dir_name+'/test_'+d['name'][i]
                +'_'+str(d['size'][i])+'.py > '+path+dir_name+'/out_test_'
                +d['name'][i]+'_'+str(d['size'][i])+'.out &'+"')\n")
            
        f2.write("# Number of processes\n")
        f2.write("N="+str(len(d['segment']))+"\n\n")
        f2.write("ProcessLimit=$(($(grep -c ^processor /proc/cpuinfo)/4))\n")
        f2.write("jobArray=()\n")
        f2.write("countOurActiveJobs(){\n")
        f2.write("    activeid=$(pgrep -u $USER)\n")
        f2.write("    jobsN=0\n")
        f2.write("    for i in ${jobArray[*]}; do  \n")
        f2.write("        for j in ${activeid[*]}; do  \n")     
        f2.write("            if [ $i = $j ]; then\n")
        f2.write("                jobsN=$(($jobsN+1))\n")
        f2.write("            fi\n")
        f2.write("        done\n")
        f2.write("    done\n")
        f2.write("}\n\n")
        f2.write("eval ProcessList=({1..$N})\n")
        f2.write("for (( k = 0; k < ${#commands[@]} ; k++ ))\n")
        f2.write("do\n")
        f2.write("    countOurActiveJobs\n")
        f2.write("    echo \"Number of jobs running: $jobsN\"\n")
        f2.write("    while [ $jobsN -ge $ProcessLimit ]\n")
        f2.write("    do\n")
        f2.write("        echo \"Waiting for space\"\n")
        f2.write("        sleep 10\n")
        f2.write("        countOurActiveJobs\n")
        f2.write("    done\n")
        f2.write("    eval ${commands[$k]}\n")
        f2.write("    jobArray+=$(echo \"$! \")\n")
        f2.write("done\n\n")
        
        f2.write("countOurActiveJobs\n")
        f2.write("while [ $jobsN != 0 ]\n")
        f2.write("do\n")
        f2.write("    echo \"Genrating... \"\n")
        f2.write("    sleep 60\n")
        f2.write("    countOurActiveJobs\n")
        f2.write("done\n")
        f2.write("nohup python "+path+dir_name+"/finalise_test.py > "
                 +path+dir_name+"/finalise_test.out")

    with open(path+dir_name+'/'+name+'_info.txt','w') as f3:
        f3.write('INFO ABOUT TEST DATA \n\n')
        f3.write('fs: '+str(fs)+'\n')
        f3.write('duration: '+str(duration)+'\n')
        f3.write('window: '+str(windowSize)+'\n')
        f3.write('timeSlides: '+str(timeSlides)+'\n'+'\n')

        for i in range(len(d['segment'])):
            f3.write(d['segment'][i][0]+' '+d['segment'][i][1]
                     +' '+str(d['size'][i])+' '
                     +str(d['start_point'][i])+'_'+d['name'][i]+'\n')
            
    with open(path+dir_name+'/finalise_test.py','w') as f4:
        f4.write("#! /usr/bin/env python3\n")
        pwd=os.getcwd()
        if 'vasileios.skliris' in pwd:
            f4.write("import sys \n")
            f4.write("sys.path.append('/home/vasileios.skliris/mly/')\n")
        f4.write("from mly.datatools.validators import *\n")
        f4.write("finalise_far('"+path+dir_name+"')\n")

    print('All set. Initiate dataset generation y/n?')
    answer4=input()

    if answer4 in ['yes','y','YES','Yes','Y']:
        os.system('nohup sh '+path+dir_name+'/auto_far.sh > '+path+dir_name+'/auto_far.out &')
        return
    else:
        print('Data generation canceled')
        os.system('cd')
        os.system('rm -r '+path+dir_name)
        return


    
def finalise_far(path):
    
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
    if len(farTests)==len(pyScripts):
        print('Files succesfully generated, all files are here')
        print(len(farTests),' out of ',len(pyScripts))
        merging_flag=True  # Declaring that merging can happen now
    
    # If some files haven't been generated it will show a failing message
    # with the processes that failed
    else:
        failed_pyScripts=[]
        for i in range(len(pyScripts)):
            pyScripts_id=pyScripts[i][5:-3]
            counter=0
            for farTest in farTests:
                if pyScripts_id in farTest:
                    counter=1
            if counter==0:
                print(pyScripts[i],' failed to proceed')
                failed_pyScripts.append(pyScripts[i])
                
                
    if merging_flag==False:
        
        if os.path.isfile(path+'/'+'auto_far_redo.sh'):
            print("\nThe following scripts failed to run trough:\n")
            for failed_pyScript in failed_pyScripts:
                print(failed_pyScript+"\n")
        else:
            with open(path+'/'+'auto_far_redo.sh','w') as f2:

                f2.write('#!/usr/bin/bash +x\n\n')
                f2.write("commands=()\n\n")

                for script in failed_pyScripts:

                    f2.write("commands+=('"+'nohup python '+path+script
                             +' > '+path+'out_'+script[:-3]+'.out &'+"')\n")

                f2.write("# Number of processes\n")
                f2.write("N="+str(len(failed_pyScripts))+"\n\n")
                f2.write("ProcessLimit=$(($(grep -c ^processor /proc/cpuinfo)/4))\n")
                f2.write("jobArray=()\n")
                f2.write("countOurActiveJobs(){\n")
                f2.write("    activeid=$(pgrep -u $USER)\n")
                f2.write("    jobsN=0\n")
                f2.write("    for i in ${jobArray[*]}; do  \n")
                f2.write("        for j in ${activeid[*]}; do  \n")     
                f2.write("            if [ $i = $j ]; then\n")
                f2.write("                jobsN=$(($jobsN+1))\n")
                f2.write("            fi\n")
                f2.write("        done\n")
                f2.write("    done\n")
                f2.write("}\n\n")
                f2.write("eval ProcessList=({1..$N})\n")
                f2.write("for (( k = 0; k < ${#commands[@]} ; k++ ))\n")
                f2.write("do\n")
                f2.write("    countOurActiveJobs\n")
                f2.write("    echo \"Number of jobs running: $jobsN\"\n")
                f2.write("    while [ jobsN -ge ProcessLimit ]\n")
                f2.write("    do\n")
                f2.write("        echo \"Waiting for space\"\n")
                f2.write("        sleep 10\n")
                f2.write("        countOurActiveJobs\n")
                f2.write("    done\n")
                f2.write("    eval ${commands[$k]}\n")
                f2.write("    jobArray+=$(echo \"$! \")\n")
                f2.write("done\n\n")

                f2.write("countOurActiveJobs\n")
                f2.write("while [ $jobsN != 0 ]\n")
                f2.write("do\n")
                f2.write("    echo \"Genrating... \"\n")
                f2.write("    sleep 60\n")
                f2.write("    countOurActiveJobs\n")
                f2.write("done\n")
                f2.write("nohup python "+path+"finalise_test.py > "+path+"finalise_test.out")

            os.system('nohup sh '+path+'auto_far_redo.sh > '+path+'auto_far_redo.out &')
            

    if merging_flag==True:
        
        setNames=[]
        setIDs=[]
        setSizes=[]
        finalNames=[]
        IDs,new_dat=[],[]
        for k in range(len(farTests)):
            if k==0:
                with open(path+farTests[k],'rb') as obj:
                    finaltest = pickle.load(obj)
                print(k,0)

            else:
                with open(path+farTests[k],'rb') as obj:
                    part_of_test = pickle.load(obj)
                finaltest = finaltest.append(part_of_test)
                print(k)

        finaltest = finaltest.sort_values(by=['SCORE'],ascending = False)
        with open(path+'FAR_TEST.pkl', 'wb') as output:
            pickle.dump(finaltest, output, pickle.HIGHEST_PROTOCOL)
                

        # Deleting unnescesary file in the folder
        for file in dirlist(path):
            if (('.out' in file) or ('.py' in file)
                or ('part_of' in file) or ('.sh' in file) or ('10000' in file)):
                os.system('rm '+path+file)


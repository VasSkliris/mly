import matplotlib as mpl
mpl.use('Agg')

from math import ceil , isnan
import os
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

from scipy.signal import spectrogram
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict    

from .simulateddetectornoise import *
from .__init__ import *

null_path=nullpath()
            
################################################################################
################################################################################
##                       #######################################################
##   HELPING FUNCTIONS   #######################################################
##                       #######################################################  


# Calculation of SNR based on the data, an injection and the background
def SNR(data,template,back=None,maximum=True,fs=2048):
    
    data=np.array(data)
    template=np.array(template)
    #back=np.array(background)

    if data.size > template.size:
        zero_pad = np.zeros(data.size - template.size)
        template=np.append(template,zero_pad)
    
    if back is None:
        back=data[-4*fs:]
    else:
        back=np.array(back)
    
    data_fft=np.fft.fft(data)
    template_fft = np.fft.fft(template)

    # -- Calculate the PSD of the data
    power_data, freq_psd = plt.psd(back, Fs=fs, NFFT=fs, visible=False)

    # -- Interpolate to get the PSD values at the needed frequencies
    datafreq = np.fft.fftfreq(data.size)*fs
    power_vec = np.interp(datafreq, freq_psd, power_data)
    
    # -- Calculate the matched filter output
    optimal = data_fft * template_fft.conjugate() / power_vec
    optimal_time = 2*np.fft.ifft(optimal)
    
    # -- Normalize the matched filter output
    df = np.abs(datafreq[1] - datafreq[0])
    sigmasq = 2*(template_fft * template_fft.conjugate() / power_vec).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = abs(optimal_time) / (sigma)
    
    
    if maximum==True:
        return(max(SNR))
    else:
        return(SNR)
    
# Creates a list of names of the files inside a folder   
def dirlist(filename):                         
    fn=os.listdir(filename)      
    fn_clean=[]
    for i in range(0,len(fn)): 
        if fn[i][0]!='.':
            fn_clean.append(fn[i])
    fn_clean.sort()
    return fn_clean

#  Loading the file with the template of the signal,
#  which is projected for the detector
def load_inj(dataset,name,detector):    
    inj=[]
    f_inj = open(null_path+'/injections/'+dataset+'/'+detector+'/'+name,'r') 
    
    for line in f_inj:              
        inj.append(float(line))     
    f_inj.close()
    return np.array(inj)

#  Loading the file with the real noise segment.
# ind is an array or list
def load_noise(fs,date_file,detector,name,ind='all'): 
    noise=[]
    with open(null_path+'/ligo_data/'+str(fs)+'/'+date_file+'/'+detector+'/'
              +name,'r') as f:
        if isinstance(ind,str) and ind=='all':
            for line in f: noise.append(float(line))
        else:
            for i in range(0,ind[0]):
                next(f)
            for i in ind:
                noise.append(float(next(f)))

    return np.array(noise)


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
    

# This function takes already made datasets and fuses them together into a new 
# one, you can either save it or just append it to a variable. Takes as input 
# the datasets name [, , , ...] files and combines them to one new.

def data_fusion(names     
                ,sizes=None
                ,save=False
                ,data_source_file = null_path+'/datasets/'):
    
    
    XS,YS=[],[]
    for name in names:
        data = io.loadmat(data_source_file+name)
        X = data['data']
        Y = data['labels']
        print('Loading file ...'+name+' with data shape:  ',X.shape)
        XS.append(X)
        YS.append(Y)
        
    if sizes==None:
        data=np.vstack(XS[:])
        labels=np.vstack(YS[:])
    elif len(sizes)==len(names):
        data=np.array(XS[0][:sizes[0]])
        labels=np.array(YS[0][:sizes[0]])
        for i in np.arange(1,len(sizes)):
            data=np.vstack((data,np.array(XS[i][:sizes[i]])))
            labels=np.vstack((labels,np.array(YS[i][:sizes[i]])))
    print(data.shape)
        
    s=np.arange(data.shape[0])
    np.random.shuffle(s)
    
    data=data[s]
    labels=labels[s]
    print('Files were fused with data shape:  ',data.shape , labels.shape)
    
    if isinstance(save,str):
        d={'data': data,'labels': labels}             
        io.savemat(save,d)
        print('File '+save+' was created')
        return
    else:
        return(data, labels)  
    
    
    
################################################################################
################################################################################
##                          ####################################################
##   GENERATING FUNCTIONS   ####################################################
##                          ####################################################



################################################################################
#################### DOCUMENTATION OF data_generator_cbc## #####################
################################################################################
#                                                                              #
# parameters       (list) A list with three elemntets: [ source_file, noise_ty #
#                  pe ,SNR ]. The source_cbcs is the name of the file containi #
#                  ng the injections we want. Noise_types are:                 #
#                                                                              #
#    'optimal'     Generated following the curve of ligo and virgo and followi #
#                  ng simulateddetectornoise.py                                #
#    'sudo_real'   Generated as optimal but the PSD curve is from real data PS #
#                  D.                                                          #
#    'real'        Real noise from the detectors.                              #
#                                                                              #
# length:          (float) The length in seconds of the generated instantiatio #
#                  ns of data.                                                 #
#                                                                              #
# fs:              (int) The sample frequency of the data. Use powers of 2 for #
#                  faster calculations.                                        #
#                                                                              #
# size:            (int) The amound of instantiations you want to generate. Po #
#                  wers of are more convinient.                                #
#                                                                              #
# detectors:       (string) A string with the initial letter of the detector y #
#                  ou want to include. H for LIGO Hanford, L for LIGO Livingst #
#                  on, V for Virgo and K for KAGRA. Example: 'HLV' or 'HLVK' o # 
#                  r 'LV'.                                                     #
#                                                                              # 
# spec:            (optional/boolean): The option to also generate spectrogram #
#                  s. Default is false. If true it will generate a separate da #
#                  taset with pictures of spectrograms of size related to the  #
#                  variable res below.                                         #
#                                                                              #
# phase:           (optional/boolean): Additionaly to spectrograms you can gen #
#                  erate phasegrams. Default is false. If true it will generat #
#                  e an additional picture under the spectrograms with the sam #
#                  e size. The size will be the same as spectrogram.           #
#                                                                              #
# res:             NEED MORE INFO HERE.                                        #
#                                                                              #
# noise_file:      (optional if noise_type is 'optimal'/ [str,str]) The name o #
#                  f the real data file you want to use as source. The path is #
#                  setted in load_noise function on emily.py. THIS HAS TO BE   #
#                  FIXED LATER!                                                #
#                  The list includes ['date_folder', 'name_of_file']. Date_fol #
#                  der is inside ligo data and contains segments of that day w #
#                  ith the name witch should be the name_of_file.              #
#                                                                              #
# t:               (optinal except psd_mode='default/ float)The duration of th # 
#                  e envelope of every instantiation used to generate the psd. # 
#                  It is used in psd_mode='default'. Prefered to be integral o #
#                  f power of 2. Default is 32.                                #
#                                                                              #
# lags:            (odd int)The number of instantiations we will use for one b #
#                  atch. To make our noise instanstiation indipendent we need  #
#                  a specific number given the detectors. For two detectors gi #
#                  ves n-1 andf or three n(n-1) indipendent instantiations.    #
#                                                                              #
# name:            (optional/string) A special tag you might want to add to yo #
#                  ur saved dataset files. Default is ''.                      #
#                                                                              #
# destination_path:(optional/string) The path where the dataset will be saved, # 
#                  the default is null_path+'/datasets/'                       #
#                                                                              #
# demo:            (optional/boolean) An option if you want to have a ploted o #
#                  ovrveiw of the data that were generated. It will not work i #
#                  n command line enviroment. Default is false.                #
################################################################################


def data_generator_cbc(parameters        
                       ,length           
                       ,fs               
                       ,size             
                       ,detectors='HLV'  
                       ,spec=True
                       ,phase=True
                       ,res=128
                       ,noise_file=None  
                       ,t=32             
                       ,lags=11
                       ,starting_point=0
                       ,name=''          
                       ,destination_path = null_path+'/datasets/cbc/'
                       ,demo=False):       

    ## INPUT CHECKING ##########
    #
    
    # parameters
    if not (isinstance(parameters[0],str) 
            and isinstance(parameters[1],str) 
            and (isinstance(parameters[2],float) 
            or isinstance(parameters[2],int)) and len(parameters)==3):
        
        raise ValueError('The parameters have to be three and in the form:' 
                         +'[list, list , float/int]')
             
    if not (os.path.isdir( null_path+'/injections/cbcs/'+parameters[0]) 
            or os.path.islink( null_path+'/injections/cbcs/'+parameters[0])):
        
        raise FileNotFoundError('No such file or directory:'
        +'\''+null_path+'/injections/cbcs/'+parameters[0]) 
    
    if (parameters[1]!='optimal' and parameters[1]!='sudo_real' 
        and parameters[1]!='real'): 
        
        raise ValueError('Wrong type of noise, the only acceptable are:'
        +'\n\'optimal\'\n\'sudo_real\'\n\'real\'')
        
    if (parameters[2]<0):
        
        raise ValueError('You cannot have a negative SNR')
    
    #length
    if (not (isinstance(length,float) or isinstance(length,int)) and length>0):
        raise ValueError('The length value has to be a possitive float'
            +' or integer.')
        
    # fs
    if not isinstance(fs,int) or fs<=0:
        raise ValueError('Sample frequency has to be a positive integer.')
    
    # detectors
    for d in detectors:
        if (d!='H' and d!='L' and d!='V' and d!='K'): 
            raise ValueError('Not acceptable character for detector.'
            +' You should include: \nH for LIGO Hanford\nL for LIGO Livingston'
            +'\nV for Virgo \nK for KAGRA\nFor example: \'HLV\', \'HLVK\'')
    # res
    if not isinstance(res,int):
        raise ValueError('Resolution variable (res) can only be integral')
    
    # noise_file    
    if ((parameters[1]=='sudo_real' or parameters[1]=='real') 
        and noise_file==None):
        
        raise TypeError('If you use suno_real or real noise you need'
            +' a real noise file as a source.')         
    if (noise_file!=None and len(noise_file)==2 and isinstance(noise_file[0],str) 
        and isinstance(noise_file[1],str)):
            
        for d in detectors:
            if os.path.isdir( null_path+'/ligo_data/'+str(fs)+'/'
                +noise_file[0]+'/'+d+'/'+noise_file[1]+'.txt'):
                
                raise FileNotFoundError('No such file or directory:'
                +'\''+null_path+'/ligo_data/'+str(fs)+'/'+noise_file[0]+'/'
                +d+'/'+noise_file[1]+'.txt\'')
    # t
    if not isinstance(t,int):
        raise ValueError('t needs to be an integral')
        
    # batch size:
    if not (isinstance(lags, int) and lags%2!=0):
        raise ValueError('lags has to be an odd integer')

    # name
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')
    
    # destination_path
    if not os.path.isdir(destination_path): 
        raise ValueError('No such path '+destination_path)
    #                        
    ########## INPUT CHECKING ## 
    

    dataset ='cbcs/'+parameters[0]
    noise_type = parameters[1]
    SNR_FIN = parameters[2]
    
    # Labels used in saving file
    #lab={10:'X', 100:'C', 1000:'M', 10000:'XM',100000:'CM'}  
    
    lab={}
    if size not in lab:
        lab[size]=str(size)

    
    # Making a list of the injection names,
    # so that we can sample randomly from them
    if 'H' in detectors: injectionH=dirlist( null_path
        +'/injections/'+dataset+'/H')
    if 'L' in detectors: injectionL=dirlist( null_path
        +'/injections/'+dataset+'/L')
    if 'V' in detectors: injectionV=dirlist( null_path
        +'/injections/'+dataset+'/V')
    
    # Integration limits for the calculation of analytical SNR
    # These values are very important for the calculation
    
    fl, fm=20, int(fs/2)
    
    # Magic number to mach the analytical computation of SNR and the
    # matched filter one. There was a mis-match which I coulndnt resolve how to
    # fix this and its not that important, if we get another nobel 
    # I will address that.
    
    magic={2048: 2**(-23./16.), 4096: 2**(-25./16.), 8192: 2**(-27./16.)}
                      

    DATA=[]
   

    ##########################
    #                        #
    # CASE OF OPTIMAL NOISE  #       
    #                        #
    ##########################
        
    if noise_type=='optimal':
        
        param=magic[fs]   
        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL))
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV))     
            
            if 'H' in detectors:
                
                # Creation of the artificial noise.
                PSDH,XH,TH=simulateddetectornoise('aligo',t,fs,10,fs/2) 
                # Calling the templates generated with PyCBC
                injH=load_inj(dataset,injectionH[inj_ind],'H')        
                # Saving the length of the injection
                inj_len=len(injH)/fs
    
                # I put a random offset for all injection so that
                # the signal is not always in the same place
                if inj_len>length: injH = injH[int(inj_len-length)*fs:]
                if inj_len<length: injH = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injH
                                ,np.zeros(int(fs*(length-inj_len)/2))))            
                if 'H' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injH[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injH[:disp]
                                ,np.zeros(int(fs*(t-length)/2)))) 
                # Making the noise a TimeSeries
                H_back=TimeSeries(XH,sample_rate=fs) 
                # Calculating the ASD so tha we can use it for whitening later
                asdH=H_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template,                
                injH_fft_0=np.fft.fft(injH)
                # we get rid of the DC value and everything above fs/2.
                injH_fft_0N=np.abs(injH_fft_0[1:int(t*fs/2)+1]) 
                                                                          
                SNR0H=np.sqrt(param*2*(1/t)
                     *np.sum(np.abs(injH_fft_0N*injH_fft_0N.conjugate())
                    [t*fl-1:t*fm-1]/PSDH[t*fl-1:t*fm-1]))
                
                
            if 'L' in detectors:
                
                # Creation of the artificial noise.
                PSDL,XL,TL=simulateddetectornoise('aligo',t,fs,10,fs/2) 
                # Calling the templates generated with PyCBC
                injL=load_inj(dataset,injectionL[inj_ind],'L')        
                # Saving the length of the injection
                inj_len=len(injL)/fs
                
                # I put a random offset for all injection
                # so that the signal is not always in the same place.
                if inj_len>length: injL = injL[int(inj_len-length)*fs:]
                if inj_len<length: injL = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injL
                                ,np.zeros(int(fs*(length-inj_len)/2))))           
                if 'L' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injL[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injL[:disp]
                                ,np.zeros(int(fs*(t-length)/2)))) 
                # Making the noise a TimeSeries
                L_back=TimeSeries(XL,sample_rate=fs) 
                # Calculating the ASD so tha we can use it for whitening later
                asdL=L_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template,                
                injL_fft_0=np.fft.fft(injL)
                # we get rid of the DC value and everything above fs/2.
                injL_fft_0N=np.abs(injL_fft_0[1:int(t*fs/2)+1]) 
                                                                          
                SNR0L=np.sqrt(param*2*(1/t)
                     *np.sum(np.abs(injL_fft_0N*injL_fft_0N.conjugate())
                    [t*fl-1:t*fm-1]/PSDL[t*fl-1:t*fm-1]))

            if 'V' in detectors:
                
                # Creation of the artificial noise.
                PSDV,XV,TV=simulateddetectornoise('aligo',t,fs,10,fs/2) 
                # Calling the templates generated with PyCBC
                injV=load_inj(dataset,injectionV[inj_ind],'V')        
                # Saving the length of the injection
                inj_len=len(injV)/fs
                
                # I put a random offset for all injection
                # so that the signal is not always in the same place.
                if inj_len>length: injV = injV[int(inj_len-length)*fs:]
                if inj_len<length: injV = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injV
                                ,np.zeros(int(fs*(length-inj_len)/2))))
                if 'V' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injV[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injV[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))  
                # Making the noise a TimeSeries
                V_back=TimeSeries(XV,sample_rate=fs) 
                # Calculating the ASD so tha we can use it for whitening later
                asdV=V_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template,                
                injV_fft_0=np.fft.fft(injV)
                # we get rid of the DC value and everything above fs/2.
                injV_fft_0N=np.abs(injV_fft_0[1:int(t*fs/2)+1]) 
                                                                          
                SNR0V=np.sqrt(param*2*(1/t)
                     *np.sum(np.abs(injV_fft_0N*injV_fft_0N.conjugate())
                    [t*fl-1:t*fm-1]/PSDV[t*fl-1:t*fm-1]))

            # Calculation of combined SNR    
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)
            
            
        # Tuning injection amplitude to the SNR wanted
        
            if 'H' in detectors:
                
                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))
                HF=TimeSeries(XH+injH_cal,sample_rate=fs,t0=0)
                #Whitening final data
                h=HF.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            if 'L' in detectors:
                
                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))
                LF=TimeSeries(XL+injL_cal,sample_rate=fs,t0=0)
                #Whitening final data
                l=LF.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            if 'V' in detectors:

                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0         
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))
                VF=TimeSeries(XV+injV_cal,sample_rate=fs,t0=0)
                #Whitening final data
                v=VF.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print('CBC',i) 
            
            

    ############################
    #                          #
    # CASE OF SUDO-REAL NOISE  #       
    #                          #
    ############################
            
    if noise_type=='sudo_real':
        
        param=1
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0]
                                                   ,'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0]
                                                   ,'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0]
                                                   ,'V',noise_file[1])
        
        ind=index_combinations(detectors = detectors
                               ,lags = lags
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)
        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL)) 
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV))
                
            if 'H' in detectors:
                # Calling the real noise segments
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]  
                # Generating the PSD of it
                p, f = plt.psd(noiseH, Fs=fs, NFFT=fs, visible=False) 
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDH,XH,TH=simulateddetectornoise([f,p],t,fs,10,fs/2)  
                # Calling the templates generated with PyCBC
                injH=load_inj(dataset,injectionH[inj_ind],'H')
                # Saving the length of the injection
                inj_len=len(injH)/fs                                  
                
                # I put a random offset for all injection, so that
                # the signal is not always in the same place
                if inj_len>length: injH = injH[int(inj_len-length)*fs:]
                if inj_len<length: injH = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injH
                                ,np.zeros(int(fs*(length-inj_len)/2))))            
                if 'H' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injH[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injH[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))  
                # Making the noise a TimeSeries
                H_back=TimeSeries(XH,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdH=H_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injH_fft_0=np.fft.fft(injH)                       
                injH_fft_0N=np.abs(injH_fft_0[1:int(t*fs/2)+1]) 
                
                SNR0H=np.sqrt(param*2*(1/t)*np.sum(np.abs(injH_fft_0N
                         *injH_fft_0N.conjugate())[t*fl-1
                            :t*fm-1]/PSDH[t*fl-1:t*fm-1]))

            if 'L' in detectors:
                

                # Calling the real noise segments
                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]  
                # Generating the PSD of it
                p, f = plt.psd(noiseL, Fs=fs, NFFT=fs, visible=False) 
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDL,XL,TL=simulateddetectornoise([f,p],t,fs,10,fs/2)  
                # Calling the templates generated with PyCBC
                injL=load_inj(dataset,injectionL[inj_ind],'L')
                # Saving the length of the injection
                inj_len=len(injL)/fs   
                
                # I put a random offset for all injection, so that
                # the signal is not always in the same place
                if inj_len>length: injL = injL[int(inj_len-length)*fs:]
                if inj_len<length: injL = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injL
                                ,np.zeros(int(fs*(length-inj_len)/2))))           
                if 'L' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injL[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injL[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))  
                # Making the noise a TimeSeries
                L_back=TimeSeries(XL,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdL=L_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injL_fft_0=np.fft.fft(injL)                       
                injL_fft_0N=np.abs(injL_fft_0[1:int(t*fs/2)+1]) 
                
                SNR0L=np.sqrt(param*2*(1/t)*np.sum(np.abs(injL_fft_0N
                         *injL_fft_0N.conjugate())[t*fl-1:t*fm-1]
                             /PSDL[t*fl-1:t*fm-1]))
                
                
            if 'V' in detectors:
                
                # Calling the real noise segments
                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]  
                # Generating the PSD of it
                p, f = plt.psd(noiseV, Fs=fs, NFFT=fs, visible=False) 
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDV,XV,TV=simulateddetectornoise([f,p],t,fs,10,fs/2)  
                # Calling the templates generated with PyCBC
                injV=load_inj(dataset,injectionV[inj_ind],'V')
                # Saving the length of the injection
                inj_len=len(injV)/fs   
                
                # I put a random offset for all injection, so that
                # the signal is not always in the same place
                if inj_len>length: injV = injV[int(inj_len-length)*fs:]
                if inj_len<length: injV = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injV
                                ,np.zeros(int(fs*(length-inj_len)/2))))
                if 'V' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injV[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injV[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))     
                # Making the noise a TimeSeries
                V_back=TimeSeries(XV,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdV=V_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injV_fft_0=np.fft.fft(injV)                       
                injV_fft_0N=np.abs(injV_fft_0[1:int(t*fs/2)+1]) 
                
                SNR0V=np.sqrt(param*2*(1/t)*np.sum(np.abs(injV_fft_0N
                         *injV_fft_0N.conjugate())[t*fl-1:t*fm-1]
                             /PSDV[t*fl-1:t*fm-1]))
                

                
            # Calculation of combined SNR   
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)

            if 'H' in detectors:
                # Tuning injection amplitude to the SNR wanted
                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))
                HF=TimeSeries(XH+injH_cal,sample_rate=fs,t0=0)
                #Whitening final data
                h=HF.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 
                
            if 'L' in detectors:
                # Tuning injection amplitude to the SNR wanted
                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))
                LF=TimeSeries(XL+injL_cal,sample_rate=fs,t0=0)
                #Whitening final data
                l=LF.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            if 'V' in detectors:
                # Tuning injection amplitude to the SNR wanted
                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0        
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))
                VF=TimeSeries(XV+injV_cal,sample_rate=fs,t0=0)
                #Whitening final data
                v=VF.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print('CBC',i)
            

    #######################
    #                     #
    # CASE OF REAL NOISE  #       
    #                     #
    #######################

    if noise_type=='real':
        
        param=1
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0]
                                                   ,'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0]
                                                   ,'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0]
                                                   ,'V',noise_file[1])
       
        ind=index_combinations(detectors = detectors
                               ,lags = lags
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)



        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL)) 
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV)) 
                
            if 'H' in detectors:
                
                # Calling the real noise segments
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]  
                # Calculatint the psd of FFT=1s
                p, f = plt.psd(noiseH, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSDH=psd_int(np.arange(0,fs/2,1/t))
                # Calling the templates generated with PyCBC
                injH=load_inj(dataset,injectionH[inj_ind],'H')
                # Saving the length of the injection
                inj_len=len(injH)/fs                                  
                
                # I put a random offset for all injection so that 
                # the signal is not always in the same place.
                if inj_len>length: injH = injH[int(inj_len-length)*fs:]
                if inj_len<length: injH = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injH
                                ,np.zeros(int(fs*(length-inj_len)/2))))            
                if 'H' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injH[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injH[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))    
                # Making the noise a TimeSeries
                H_back=TimeSeries(noiseH,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdH=H_back.asd(1,0.5)                                    

                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injH_fft_0=np.fft.fft(injH)                       
                injH_fft_0N=np.abs(injH_fft_0[1:int(t*fs/2)+1])  
                
                SNR0H=np.sqrt(param*2*(1/t)*np.sum(np.abs(injH_fft_0N
                        *injH_fft_0N.conjugate())[t*fl-1:t*fm-1]
                                /PSDH[t*fl-1:t*fm-1]))
                
            if 'L' in detectors:
                
                # Calling the real noise segments
                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]  
                # Calculatint the psd of FFT=1s
                p, f = plt.psd(noiseL, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSDL=psd_int(np.arange(0,fs/2,1/t))
                # Calling the templates generated with PyCBC
                injL=load_inj(dataset,injectionL[inj_ind],'L')
                # Saving the length of the injection
                inj_len=len(injL)/fs  
                
                # I put a random offset for all injection so that
                # the signal is not always in the same place.
                if inj_len>length: injL = injL[int(inj_len-length)*fs:]
                if inj_len<length: injL = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injL
                                ,np.zeros(int(fs*(length-inj_len)/2))))           
                if 'L' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injL[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injL[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))  
                # Making the noise a TimeSeries
                L_back=TimeSeries(noiseL,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdL=L_back.asd(1,0.5)                                    

                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injL_fft_0=np.fft.fft(injL)                       
                injL_fft_0N=np.abs(injL_fft_0[1:int(t*fs/2)+1])
                
                SNR0L=np.sqrt(param*2*(1/t)*np.sum(np.abs(injL_fft_0N
                        *injL_fft_0N.conjugate())[t*fl-1:t*fm-1]
                                /PSDL[t*fl-1:t*fm-1]))

            if 'V' in detectors:

                # Calling the real noise segments
                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]  
                # Calculatint the psd of FFT=1s
                p, f = plt.psd(noiseV, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSDV=psd_int(np.arange(0,fs/2,1/t))
                # Calling the templates generated with PyCBC
                injV=load_inj(dataset,injectionV[inj_ind],'V')
                # Saving the length of the injection
                inj_len=len(injV)/fs  
                
                # I put a random offset for all injection so that
                # the signal is not always in the same place.
                if 'H' not in detectors:
                    if length==inj_len:                                   
                        disp = np.random.randint(0,int(length*fs/2))      
                    elif length > inj_len:                      
                        disp = np.random.randint(-int(fs*(length-inj_len)/2)   
                                                 ,int(fs*(length-inj_len)/2)) 
                # I put a random offset for all injection so that 
                # the signal is not always in the same place.
                if inj_len>length: injV = injV[int(inj_len-length)*fs:]
                if inj_len<length: injV = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injV
                                ,np.zeros(int(fs*(length-inj_len)/2))))
                if 'V' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injV[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injV[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))    
                # Making the noise a TimeSeries
                V_back=TimeSeries(noiseV,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdV=V_back.asd(1,0.5)                                    

                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injV_fft_0=np.fft.fft(injV)                       
                injV_fft_0N=np.abs(injV_fft_0[1:int(t*fs/2)+1])
                
                SNR0V=np.sqrt(param*2*(1/t)*np.sum(np.abs(injV_fft_0N
                        *injV_fft_0N.conjugate())[t*fl-1:t*fm-1]
                                /PSDV[t*fl-1:t*fm-1]))

                
                
            # Calculation of combined SNR
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)
            
            if 'H' in detectors:

                # Tuning injection amplitude to the SNR wanted
                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))
                HF=TimeSeries(noiseH+injH_cal,sample_rate=fs,t0=0)
                #Whitening final data
                h=HF.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)]

            if 'L' in detectors:
                
                # Tuning injection amplitude to the SNR wanted
                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))
                LF=TimeSeries(noiseL+injL_cal,sample_rate=fs,t0=0)
                #Whitening final data
                l=LF.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 
                
            if 'V' in detectors:
                
                # Tuning injection amplitude to the SNR wanted
                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0         
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))
                VF=TimeSeries(noiseV+injV_cal,sample_rate=fs,t0=0)
                #Whitening final data
                v=VF.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 
            

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print('CBC',i,)       
                
    # We are gonna use Keras which demands a specific format of datasets
    # So we do that formating here
    
    data=[]
    labels=[]
    for row in DATA:
        data.append(row[0])
        labels=labels+[row[1]]
        
    del DATA   
    data, labels= np.array(data), np.array(labels)
    labels=labels.reshape(1*size,1)

    
    data = data.transpose((0,2,1))
    
    print('Shape of data created:   ',data.shape)
    print('Shape of labels created: ',labels.shape)
    
    if demo==True:
        for i in range(0,10):
            plt.figure(figsize=(10,5))
            
            for j in range(0,len(detectors)):
                plt.plot(np.arange(0,length,1./fs),data[i][:,j]+j*4)
                
            plt.show()


    d={'data': data,'labels': labels}
    save_name = (detectors+'_time_cbc_with_'+noise_type+'_noise_SNR'
                 +str(SNR_FIN)+'_'+name+lab[size])
    
    print(destination_path+save_name)
    io.savemat(destination_path+save_name,d)
    print('File '+save_name+'.mat was created')
    
    if spec==True:
        
        data_spec=[]

        stride ,fft_l = res/fs, res/fs  

        for time_series in data:
            spec=[]
            for i in range(0,len(detectors)):
                f, t, spec_gram = spectrogram(time_series[:,i]
                            , fs 
                            , window='hanning'
                            , nperseg=int(stride*fs)
                            , noverlap=0
                            , nfft =int(fft_l*fs)
                            , mode='complex'
                            ,scaling ='density')
                spectogram=np.array(np.abs(spec_gram))
                if phase==True: phasegram=np.array(np.angle(spec_gram))
                spec.append(spectogram)
                if phase==True: spec.append(phasegram)

            data_spec.append(spec)

        data_spec=np.array(data_spec)

        data_spec = data_spec.transpose((0,2,3,1))
        #data_spec = np.flip(data_spec,1)    

        print('Shape of data created:   ',data_spec.shape)
        print('Shape of labels created: ',labels.shape)

        d={'data': data_spec,'labels': labels}        
        save_name = detectors+'_spec_cbc_with_'+noise_type+'_noise_SNR'
        +str(SNR_FIN)+'_'+name+lab[size]
        
        io.savemat(destination_path+save_name,d)
        print('File '+save_name+'.mat was created')
    
        if demo==True:
            for i in range(0,10):
                if len(detectors)==3:
                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,::2]
                                   /np.max(data_spec[i][:,:,::2])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1::2]
                                   /np.max(data_spec[i][:,:,1::2])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i]/np.max(data_spec[i])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                elif len(detectors) in [1,2]:

                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,0]
                                   /np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1]
                                   /np.max(data_spec[i][:,:,1])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i][:,:,1]
                                   /np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')


            plt.show()


################################################################################
#################### DOCUMENTATION OF data_generator_inj #######################
################################################################################
#                                                                              #
# parameters       (list) A list with three elemntets: [ source_file, noise_ty #
#                  pe ,SNR ]. The source_cbcs is the name of the file containi #
#                  ng the injections we want. Noise_types are:                 #
#                                                                              #
#    'optimal'     Generated following the curve of ligo and virgo and followi #
#                  ng simulateddetectornoise.py                                #
#    'sudo_real'   Generated as optimal but the PSD curve is from real data PS #
#                  D.                                                          #
#    'real'        Real noise from the detectors.                              #
#                                                                              #
# length:          (float) The length in seconds of the generated instantiatio #
#                  ns of data.                                                 #
#                                                                              #
# fs:              (int) The sample frequency of the data. Use powers of 2 for #
#                  faster calculations.                                        #
#                                                                              #
# size:            (int) The amound of instantiations you want to generate. Po #
#                  wers of are more convinient.                                #
#                                                                              #
# detectors:       (string) A string with the initial letter of the detector y #
#                  ou want to include. H for LIGO Hanford, L for LIGO Livingst #
#                  on, V for Virgo and K for KAGRA. Example: 'HLV' or 'HLVK' o # 
#                  r 'LV'.                                                     #
#                                                                              # 
# spec:            (optional/boolean): The option to also generate spectrogram #
#                  s. Default is false. If true it will generate a separate da #
#                  taset with pictures of spectrograms of size related to the  #
#                  variable res below.                                         #
#                                                                              #
# phase:           (optional/boolean): Additionaly to spectrograms you can gen #
#                  erate phasegrams. Default is false. If true it will generat #
#                  e an additional picture under the spectrograms with the sam #
#                  e size. The size will be the same as spectrogram.           #
#                                                                              #
# res:             NEED MORE INFO HERE.                                        #
#                                                                              #
# noise_file:      (optional if noise_type is 'optimal'/ [str,str]) The name o #
#                  f the real data file you want to use as source. The path is #
#                  setted in load_noise function on emily.py. THIS HAS TO BE   #
#                  FIXED LATER!                                                #
#                  The list includes ['date_folder', 'name_of_file']. Date_fol #
#                  der is inside ligo data and contains segments of that day w #
#                  ith the name witch should be the name_of_file.              #
#                                                                              #
# t:               (optinal except psd_mode='default/ float)The duration of th # 
#                  e envelope of every instantiation used to generate the psd. # 
#                  It is used in psd_mode='default'. Prefered to be integral o #
#                  f power of 2. Default is 32.                                #
#                                                                              #
# lags:            (odd int)The number of instantiations we will use for one b #
#                  atch. To make our noise instanstiation indipendent we need  #
#                  a specific number given the detectors. For two detectors gi #
#                  ves n-1 andf or three n(n-1) indipendent instantiations.    #
#                                                                              #
# name:            (optional/string) A special tag you might want to add to yo #
#                  ur saved dataset files. Default is ''.                      #
#                                                                              #
# destination_path:(optional/string) The path where the dataset will be saved, # 
#                  the default is null_path+'/datasets/'                       #
#                                                                              #
# demo:            (optional/boolean) An option if you want to have a ploted o #
#                  ovrveiw of the data that were generated. It will not work i #
#                  n command line enviroment. Default is false.                #
################################################################################

def data_generator_inj(parameters        
                       ,length           
                       ,fs               
                       ,size 
                       ,spec=False
                       ,phase=False 
                       ,detectors='HLV'  
                       ,res=128
                       ,noise_file=None  
                       ,t=32             
                       ,lags=11
                       ,starting_point=0
                       ,name=''          
                       ,destination_path = null_path+'/datasets'
                       ,demo=False):   
    
    ## INPUT CHECKING ##########
    #
    
    # parameters
    if not (isinstance(parameters[0],str) 
            and isinstance(parameters[1],str) 
            and (isinstance(parameters[2],float) 
            or isinstance(parameters[2],int)) and len(parameters)==3):
        
        raise ValueError('The parameters have to be three and in the form:' 
                         +'[list, list , float/int]')
             
    if not (os.path.isdir( null_path+'/injections/'+parameters[0]) 
            or os.path.islink( null_path+'/injections/'+parameters[0])):
        
        raise FileNotFoundError('No such file or directory:'
        +'\''+null_path+'/injections/'+parameters[0]) 
    
    if (parameters[1]!='optimal' and parameters[1]!='sudo_real' 
        and parameters[1]!='real'): 
        
        raise ValueError('Wrong type of noise, the only acceptable are:'
        +'\n\'optimal\'\n\'sudo_real\'\n\'real\'')
        
    if (parameters[2]<0):
        
        raise ValueError('You cannot have a negative SNR')
    
    #length
    if (not (isinstance(length,float) or isinstance(length,int)) and length>0):
        raise ValueError('The length value has to be a possitive float'
            +' or integer.')
        
    # fs
    if not isinstance(fs,int) or fs<=0:
        raise ValueError('Sample frequency has to be a positive integer.')
    
    # detectors
    for d in detectors:
        if (d!='H' and d!='L' and d!='V' and d!='K'): 
            raise ValueError('Not acceptable character for detector.'
            +' You should include: \nH for LIGO Hanford\nL for LIGO Livingston'
            +'\nV for Virgo \nK for KAGRA\nFor example: \'HLV\', \'HLVK\'')
    # res
    if not isinstance(res,int):
        raise ValueError('Resolution variable (res) can only be integral')
    
    # noise_file    
    if ((parameters[1]=='sudo_real' or parameters[1]=='real') 
        and noise_file==None):
        
        raise TypeError('If you use suno_real or real noise you need'
            +' a real noise file as a source.')         
    if (noise_file!=None and len(noise_file)==2 and isinstance(noise_file[0],str) 
        and isinstance(noise_file[1],str)):
            
        for d in detectors:
            if os.path.isdir( null_path+'/ligo_data/'+str(fs)+'/'
                +noise_file[0]+'/'+d+'/'+noise_file[1]+'.txt'):
                
                raise FileNotFoundError('No such file or directory:'
                +'\''+null_path+'/ligo_data/'+str(fs)+'/'+noise_file[0]+'/'
                +d+'/'+noise_file[1]+'.txt\'')
    # t
    if not isinstance(t,int):
        raise ValueError('t needs to be an integral')
        
    # batch size:
    if not (isinstance(lags, int) and lags%2!=0):
        raise ValueError('lags has to be an odd integer')

    # name
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')
    
    # destination_path
    if not os.path.isdir(destination_path): 
        raise ValueError('No such path '+destination_path)
    #                        
    ########## INPUT CHECKING ## 
    

    dataset =parameters[0]
    noise_type = parameters[1]
    SNR_FIN = parameters[2]
    
    inj_type = dataset.split('/')[1].split('_')[0]
    
    lab={}
    if size not in lab:
        lab[size]=str(size)

    # Making a list of the injection names,
    # so that we can sample randomly from them
    if 'H' in detectors: injectionH=dirlist( null_path
        +'/injections/'+dataset+'/H')
    if 'L' in detectors: injectionL=dirlist( null_path
        +'/injections/'+dataset+'/L')
    if 'V' in detectors: injectionV=dirlist( null_path
        +'/injections/'+dataset+'/V')
    
    # Integration limits for the calculation of analytical SNR
    # These values are very important for the calculation
    fl, fm=20, int(fs/2)
    
    # Magic number to mach the analytical computation of SNR and the
    # matched filter one. There was a mis-match which I coulndnt resolve how to
    # fix this and its not that important, if we get another nobel 
    # I will address that.
    
    magic={2048: 2**(-23./16.), 4096: 2**(-25./16.), 8192: 2**(-27./16.)}
                      

    DATA=[]
   

    ##########################
    #                        #
    # CASE OF OPTIMAL NOISE  #       
    #                        #
    ##########################
        
    if noise_type=='optimal':
        
        param=magic[fs]   
        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL))
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV))     
            
            if 'H' in detectors:
                
                # Creation of the artificial noise.
                PSDH,XH,TH=simulateddetectornoise('aligo',t,fs,10,fs/2) 
                # Calling the templates generated with PyCBC
                injH=load_inj(dataset,injectionH[inj_ind],'H')        
                # Saving the length of the injection
                inj_len=len(injH)/fs
    
                # I put a random offset for all injection so that
                # the signal is not always in the same place
                if inj_len>length: injH = injH[int(inj_len-length)*fs:]
                if inj_len<length: injH = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injH
                                ,np.zeros(int(fs*(length-inj_len)/2))))            
                if 'H' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injH[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injH[:disp]
                                ,np.zeros(int(fs*(t-length)/2)))) 
                # Making the noise a TimeSeries
                H_back=TimeSeries(XH,sample_rate=fs) 
                # Calculating the ASD so tha we can use it for whitening later
                asdH=H_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template,                
                injH_fft_0=np.fft.fft(injH)
                # we get rid of the DC value and everything above fs/2.
                injH_fft_0N=np.abs(injH_fft_0[1:int(t*fs/2)+1]) 
                                                                          
                SNR0H=np.sqrt(param*2*(1/t)
                     *np.sum(np.abs(injH_fft_0N*injH_fft_0N.conjugate())
                    [t*fl-1:t*fm-1]/PSDH[t*fl-1:t*fm-1]))
                
                
            if 'L' in detectors:
                
                # Creation of the artificial noise.
                PSDL,XL,TL=simulateddetectornoise('aligo',t,fs,10,fs/2) 
                # Calling the templates generated with PyCBC
                injL=load_inj(dataset,injectionL[inj_ind],'L')        
                # Saving the length of the injection
                inj_len=len(injL)/fs
                
                # I put a random offset for all injection
                # so that the signal is not always in the same place.
                if inj_len>length: injL = injL[int(inj_len-length)*fs:]
                if inj_len<length: injL = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injL
                                ,np.zeros(int(fs*(length-inj_len)/2))))           
                if 'L' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injL[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injL[:disp]
                                ,np.zeros(int(fs*(t-length)/2)))) 
                # Making the noise a TimeSeries
                L_back=TimeSeries(XL,sample_rate=fs) 
                # Calculating the ASD so tha we can use it for whitening later
                asdL=L_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template,                
                injL_fft_0=np.fft.fft(injL)
                # we get rid of the DC value and everything above fs/2.
                injL_fft_0N=np.abs(injL_fft_0[1:int(t*fs/2)+1]) 
                                                                          
                SNR0L=np.sqrt(param*2*(1/t)
                     *np.sum(np.abs(injL_fft_0N*injL_fft_0N.conjugate())
                    [t*fl-1:t*fm-1]/PSDL[t*fl-1:t*fm-1]))

            if 'V' in detectors:
                
                # Creation of the artificial noise.
                PSDV,XV,TV=simulateddetectornoise('aligo',t,fs,10,fs/2) 
                # Calling the templates generated with PyCBC
                injV=load_inj(dataset,injectionV[inj_ind],'V')        
                # Saving the length of the injection
                inj_len=len(injV)/fs
                
                # I put a random offset for all injection
                # so that the signal is not always in the same place.
                if inj_len>length: injV = injV[int(inj_len-length)*fs:]
                if inj_len<length: injV = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injV
                                ,np.zeros(int(fs*(length-inj_len)/2))))
                if 'V' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injV[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injV[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))  
                # Making the noise a TimeSeries
                V_back=TimeSeries(XV,sample_rate=fs) 
                # Calculating the ASD so tha we can use it for whitening later
                asdV=V_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template,                
                injV_fft_0=np.fft.fft(injV)
                # we get rid of the DC value and everything above fs/2.
                injV_fft_0N=np.abs(injV_fft_0[1:int(t*fs/2)+1]) 
                                                                          
                SNR0V=np.sqrt(param*2*(1/t)
                     *np.sum(np.abs(injV_fft_0N*injV_fft_0N.conjugate())
                    [t*fl-1:t*fm-1]/PSDV[t*fl-1:t*fm-1]))

            # Calculation of combined SNR    
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)
            
            
        # Tuning injection amplitude to the SNR wanted
        
            if 'H' in detectors:
                
                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))
                HF=TimeSeries(XH+injH_cal,sample_rate=fs,t0=0)
                #Whitening final data
                h=HF.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            if 'L' in detectors:
                
                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))
                LF=TimeSeries(XL+injL_cal,sample_rate=fs,t0=0)
                #Whitening final data
                l=LF.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            if 'V' in detectors:

                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0         
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))
                VF=TimeSeries(XV+injV_cal,sample_rate=fs,t0=0)
                #Whitening final data
                v=VF.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print(inj_type.upper(),i) 
            
            

    ############################
    #                          #
    # CASE OF SUDO-REAL NOISE  #       
    #                          #
    ############################
            
    if noise_type=='sudo_real':
        
        param=1
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0]
                                                   ,'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0]
                                                   ,'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0]
                                                   ,'V',noise_file[1])
        
        ind=index_combinations(detectors = detectors
                               ,lags = lags
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)
        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL)) 
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV))
                
            if 'H' in detectors:
                # Calling the real noise segments
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]  
                # Generating the PSD of it
                p, f = plt.psd(noiseH, Fs=fs, NFFT=fs, visible=False) 
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDH,XH,TH=simulateddetectornoise([f,p],t,fs,10,fs/2)  
                # Calling the templates generated with PyCBC
                injH=load_inj(dataset,injectionH[inj_ind],'H')
                # Saving the length of the injection
                inj_len=len(injH)/fs                                  
                
                # I put a random offset for all injection, so that
                # the signal is not always in the same place
                if inj_len>length: injH = injH[int(inj_len-length)*fs:]
                if inj_len<length: injH = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injH
                                ,np.zeros(int(fs*(length-inj_len)/2))))            
                if 'H' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injH[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injH[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))  
                # Making the noise a TimeSeries
                H_back=TimeSeries(XH,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdH=H_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injH_fft_0=np.fft.fft(injH)                       
                injH_fft_0N=np.abs(injH_fft_0[1:int(t*fs/2)+1]) 
                
                SNR0H=np.sqrt(param*2*(1/t)*np.sum(np.abs(injH_fft_0N
                         *injH_fft_0N.conjugate())[t*fl-1
                            :t*fm-1]/PSDH[t*fl-1:t*fm-1]))
                
            if 'L' in detectors:
                

                # Calling the real noise segments
                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]  
                # Generating the PSD of it
                p, f = plt.psd(noiseL, Fs=fs, NFFT=fs, visible=False) 
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDL,XL,TL=simulateddetectornoise([f,p],t,fs,10,fs/2)  
                # Calling the templates generated with PyCBC
                injL=load_inj(dataset,injectionL[inj_ind],'L')
                # Saving the length of the injection
                inj_len=len(injL)/fs   
                
                # I put a random offset for all injection, so that
                # the signal is not always in the same place
                if inj_len>length: injL = injL[int(inj_len-length)*fs:]
                if inj_len<length: injL = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injL
                                ,np.zeros(int(fs*(length-inj_len)/2))))           
                if 'L' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injL[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injL[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))  
                # Making the noise a TimeSeries
                L_back=TimeSeries(XL,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdL=L_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injL_fft_0=np.fft.fft(injL)                       
                injL_fft_0N=np.abs(injL_fft_0[1:int(t*fs/2)+1]) 
                
                SNR0L=np.sqrt(param*2*(1/t)*np.sum(np.abs(injL_fft_0N
                         *injL_fft_0N.conjugate())[t*fl-1:t*fm-1]
                             /PSDL[t*fl-1:t*fm-1]))
                
                
            if 'V' in detectors:
                
                # Calling the real noise segments
                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]  
                # Generating the PSD of it
                p, f = plt.psd(noiseV, Fs=fs, NFFT=fs, visible=False) 
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDV,XV,TV=simulateddetectornoise([f,p],t,fs,10,fs/2)  
                # Calling the templates generated with PyCBC
                injV=load_inj(dataset,injectionV[inj_ind],'V')
                # Saving the length of the injection
                inj_len=len(injV)/fs   
                
                # I put a random offset for all injection, so that
                # the signal is not always in the same place
                if inj_len>length: injV = injV[int(inj_len-length)*fs:]
                if inj_len<length: injV = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injV
                                ,np.zeros(int(fs*(length-inj_len)/2))))
                if 'V' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injV[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injV[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))     
                # Making the noise a TimeSeries
                V_back=TimeSeries(XV,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdV=V_back.asd(1,0.5)                                    
                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injV_fft_0=np.fft.fft(injV)                       
                injV_fft_0N=np.abs(injV_fft_0[1:int(t*fs/2)+1]) 
                
                SNR0V=np.sqrt(param*2*(1/t)*np.sum(np.abs(injV_fft_0N
                         *injV_fft_0N.conjugate())[t*fl-1:t*fm-1]
                             /PSDV[t*fl-1:t*fm-1]))
                

                
            # Calculation of combined SNR   
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)

            if 'H' in detectors:
                # Tuning injection amplitude to the SNR wanted
                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))
                HF=TimeSeries(XH+injH_cal,sample_rate=fs,t0=0)
                #Whitening final data
                h=HF.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 
                
            if 'L' in detectors:
                # Tuning injection amplitude to the SNR wanted
                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))
                LF=TimeSeries(XL+injL_cal,sample_rate=fs,t0=0)
                #Whitening final data
                l=LF.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            if 'V' in detectors:
                # Tuning injection amplitude to the SNR wanted
                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0        
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))
                VF=TimeSeries(XV+injV_cal,sample_rate=fs,t0=0)
                #Whitening final data
                v=VF.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)] 

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print(inj_type.upper(),i)
            

    #######################
    #                     #
    # CASE OF REAL NOISE  #       
    #                     #
    #######################

    if noise_type=='real':
        
        param=1
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0]
                                                   ,'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0]
                                                   ,'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0]
                                                   ,'V',noise_file[1])
       
        ind=index_combinations(detectors = detectors
                               ,lags = lags
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)



        
        for i in range(0,size):

            if 'H' in detectors: inj_ind=np.random.randint(0,len(injectionH))  
            elif 'L' in detectors: inj_ind=np.random.randint(0,len(injectionL)) 
            elif 'V' in detectors: inj_ind=np.random.randint(0,len(injectionV)) 
                
            if 'H' in detectors:
                
                # Calling the real noise segments
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]  
                # Calculatint the psd of FFT=1s
                p, f = plt.psd(noiseH, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSDH=psd_int(np.arange(0,fs/2,1/t))
                # Calling the templates generated with PyCBC
                injH=load_inj(dataset,injectionH[inj_ind],'H')
                # Saving the length of the injection
                inj_len=len(injH)/fs                                  
                
                # I put a random offset for all injection so that 
                # the signal is not always in the same place.
                if inj_len>length: injH = injH[int(inj_len-length)*fs:]
                if inj_len<length: injH = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injH
                                ,np.zeros(int(fs*(length-inj_len)/2))))            
                if 'H' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injH[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injH = injH=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injH[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))    
                # Making the noise a TimeSeries
                H_back=TimeSeries(noiseH,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdH=H_back.asd(1,0.5)                                    

                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injH_fft_0=np.fft.fft(injH)                       
                injH_fft_0N=np.abs(injH_fft_0[1:int(t*fs/2)+1])  
                
                print(inj_len*fs,len(injH),injectionH[inj_ind], disp/fs)
                print(len((injH_fft_0N*injH_fft_0N.conjugate())[t*fl-1:t*fm-1])
                      , t
                      ,len(injH_fft_0N)
                      ,len(injH_fft_0N.conjugate())
                      , len(injH_fft_0N[t*fl-1:t*fm-1])
                      , len(PSDH[t*fl-1:t*fm-1])
                     )
                
                SNR0H=np.sqrt(param*2*(1/t)*np.sum(np.abs(injH_fft_0N
                        *injH_fft_0N.conjugate())[t*fl-1:t*fm-1]
                                /PSDH[t*fl-1:t*fm-1]))
                
            if 'L' in detectors:
                
                # Calling the real noise segments
                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]  
                # Calculatint the psd of FFT=1s
                p, f = plt.psd(noiseL, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSDL=psd_int(np.arange(0,fs/2,1/t))
                # Calling the templates generated with PyCBC
                injL=load_inj(dataset,injectionL[inj_ind],'L')
                # Saving the length of the injection
                inj_len=len(injL)/fs  
                
                # I put a random offset for all injection so that
                # the signal is not always in the same place.
                if inj_len>length: injL = injL[int(inj_len-length)*fs:]
                if inj_len<length: injL = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injL
                                ,np.zeros(int(fs*(length-inj_len)/2))))           
                if 'L' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injL[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injL = injL=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injL[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))  
                # Making the noise a TimeSeries
                L_back=TimeSeries(noiseL,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdL=L_back.asd(1,0.5)                                    

                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injL_fft_0=np.fft.fft(injL)                       
                injL_fft_0N=np.abs(injL_fft_0[1:int(t*fs/2)+1])
                
                SNR0L=np.sqrt(param*2*(1/t)*np.sum(np.abs(injL_fft_0N
                        *injL_fft_0N.conjugate())[t*fl-1:t*fm-1]
                                /PSDL[t*fl-1:t*fm-1]))

            if 'V' in detectors:

                # Calling the real noise segments
                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]  
                # Calculatint the psd of FFT=1s
                p, f = plt.psd(noiseV, Fs=fs,NFFT=fs)
                # Interpolate so that has t*fs values
                psd_int=interp1d(f,p)                                     
                PSDV=psd_int(np.arange(0,fs/2,1/t))
                # Calling the templates generated with PyCBC
                injV=load_inj(dataset,injectionV[inj_ind],'V')
                # Saving the length of the injection
                inj_len=len(injV)/fs  
                
                # I put a random offset for all injection so that
                # the signal is not always in the same place.
                if 'H' not in detectors:
                    if length==inj_len:                                   
                        disp = np.random.randint(0,int(length*fs/2))      
                    elif length > inj_len:                      
                        disp = np.random.randint(-int(fs*(length-inj_len)/2)   
                                                 ,int(fs*(length-inj_len)/2)) 
                # I put a random offset for all injection so that 
                # the signal is not always in the same place.
                if inj_len>length: injV = injV[int(inj_len-length)*fs:]
                if inj_len<length: injV = np.hstack((np.zeros(int(fs*(length
                                -inj_len)/2))
                                , injV
                                ,np.zeros(int(fs*(length-inj_len)/2))))
                if 'V' == detectors[0]:

                    disp = np.random.randint(-int(fs*(length-inj_len)/2)
                                             ,int(inj_len*fs/2))

                if disp >= 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2))
                                , injV[disp:]
                                ,np.zeros(int(fs*(t-length)/2)+disp)))

                if disp < 0: injV = injV=np.hstack((np.zeros(int(fs*(t
                                -length)/2)-disp)
                                , injV[:disp]
                                ,np.zeros(int(fs*(t-length)/2))))    
                # Making the noise a TimeSeries
                V_back=TimeSeries(noiseV,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdV=V_back.asd(1,0.5)                                    

                # Calculating the one sided fft of the template and
                # we get rid of the DC value and everything above fs/2.
                injV_fft_0=np.fft.fft(injV)                       
                injV_fft_0N=np.abs(injV_fft_0[1:int(t*fs/2)+1])
                
                SNR0V=np.sqrt(param*2*(1/t)*np.sum(np.abs(injV_fft_0N
                        *injV_fft_0N.conjugate())[t*fl-1:t*fm-1]
                                /PSDV[t*fl-1:t*fm-1]))

                
               
            # Calculation of combined SNR
            SNR0=0
            if 'H' in detectors: SNR0+=SNR0H**2
            if 'L' in detectors: SNR0+=SNR0L**2     
            if 'V' in detectors: SNR0+=SNR0V**2
            SNR0=np.sqrt(SNR0)
            
            if 'H' in detectors:

                # Tuning injection amplitude to the SNR wanted
                fftH_cal=(SNR_FIN/SNR0)*injH_fft_0         
                injH_cal=np.real(np.fft.ifft(fftH_cal*fs))
                HF=TimeSeries(noiseH+injH_cal,sample_rate=fs,t0=0)
                #Whitening final data
                h=HF.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)]

            if 'L' in detectors:
                
                # Tuning injection amplitude to the SNR wanted
                fftL_cal=(SNR_FIN/SNR0)*injL_fft_0         
                injL_cal=np.real(np.fft.ifft(fftL_cal*fs))
                LF=TimeSeries(noiseL+injL_cal,sample_rate=fs,t0=0)
                #Whitening final data
                l=LF.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)]
                
            if 'V' in detectors:
                
                # Tuning injection amplitude to the SNR wanted
                fftV_cal=(SNR_FIN/SNR0)*injV_fft_0         
                injV_cal=np.real(np.fft.ifft(fftV_cal*fs))
                VF=TimeSeries(noiseV+injV_cal,sample_rate=fs,t0=0)
                #Whitening final data
                v=VF.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                            int(((t+length)/2)*fs)]


            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))


            row=[dumie,1] # 'CHIRP' = 1
            DATA.append(row)    


            print(inj_type.upper(),i,)       
                
    # We are gonna use Keras which demands a specific format of datasets
    # So we do that formating here
    
    data=[]
    labels=[]
    for row in DATA:
        data.append(row[0])
        labels=labels+[row[1]]
        
    del DATA   
    data, labels= np.array(data), np.array(labels)
    labels=labels.reshape(1*size,1)

    
    data = data.transpose((0,2,1))
    
    print('Shape of data created:   ',data.shape)
    print('Shape of labels created: ',labels.shape)
    
    if demo==True:
        for i in range(0,10):
            plt.figure(figsize=(10,5))
            
            for j in range(0,len(detectors)):
                plt.plot(np.arange(0,length,1./fs),data[i][:,j]+j*4)
                
            plt.show()


    d={'data': data,'labels': labels}
    save_name = (detectors+'_time_'+inj_type+'_with_'+noise_type+'_noise_SNR'
                 +str(SNR_FIN)+'_'+name+lab[size])
    
    print(destination_path+save_name)
    io.savemat(destination_path+save_name,d)
    print('File '+save_name+'.mat was created')
    
    if spec==True:
        
        data_spec=[]

        stride ,fft_l = res/fs, res/fs  

        for time_series in data:
            spec=[]
            for i in range(0,len(detectors)):
                f, t, spec_gram = spectrogram(time_series[:,i]
                            , fs 
                            , window='hanning'
                            , nperseg=int(stride*fs)
                            , noverlap=0
                            , nfft =int(fft_l*fs)
                            , mode='complex'
                            ,scaling ='density')
                spectogram=np.array(np.abs(spec_gram))
                if phase==True: phasegram=np.array(np.angle(spec_gram))
                spec.append(spectogram)
                if phase==True: spec.append(phasegram)

            data_spec.append(spec)

        data_spec=np.array(data_spec)

        data_spec = data_spec.transpose((0,2,3,1))
        #data_spec = np.flip(data_spec,1)    

        print('Shape of data created:   ',data_spec.shape)
        print('Shape of labels created: ',labels.shape)

        d={'data': data_spec,'labels': labels}        
        save_name = detectors+'_spec_'+inj_type+'_with_'+noise_type+'_noise_SNR'
        +str(SNR_FIN)+'_'+name+lab[size]
        
        io.savemat(destination_path+save_name,d)
        print('File '+save_name+'.mat was created')
    
        if demo==True:
            for i in range(0,10):
                if len(detectors)==3:
                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,::2]
                                   /np.max(data_spec[i][:,:,::2])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1::2]
                                   /np.max(data_spec[i][:,:,1::2])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i]/np.max(data_spec[i])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                elif len(detectors) in [1,2]:

                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,0]
                                   /np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1]
                                   /np.max(data_spec[i][:,:,1])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i][:,:,0]
                                   /np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) 
                                            ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')


            plt.show()

################################################################################
#################### DOCUMENTATION OF data_generator_noise #####################
################################################################################
#                                                                              #
# noise_type       (string) The type of noise to be generated. There are three #
#                  options:                                                    #
#                                                                              #
#    'optimal'     Generated following the curve of ligo and virgo and followi #
#                  ng simulateddetectornoise.py                                #
#    'sudo_real'   Generated as optimal but the PSD curve is from real data PS #
#                  D.                                                          #
#    'real'        Real noise from the detectors.                              #
#                                                                              #
# length:          (float) The length in seconds of the generated instantiatio #
#                  ns of data.                                                 #
#                                                                              #
# fs:              (int) The sample frequency of the data. Use powers of 2 for #
#                  faster calculations.                                        #
#                                                                              #
# size:            (int) The amound of instantiations you want to generate. Po #
#                  wers of are more convinient.                                #
#                                                                              #
# detectors:       (string) A string with the initial letter of the detector y #
#                  ou want to include. H for LIGO Hanford, L for LIGO Livingst #
#                  on, V for Virgo and K for KAGRA. Example: 'HLV' or 'HLVK' o # 
#                  r 'LV'.                                                     #
#                                                                              #
# spec:            (optional/boolean): The option to also generate spectrogram #
#                  s. Default is false. If true it will generate a separate da #
#                  taset with pictures of spectrograms of size related to the  #
#                  variable res below.                                         #
#                                                                              #
# phase:           (optional/boolean): Additionaly to spectrograms you can gen #
#                  erate phasegrams. Default is false. If true it will generat #
#                  e an additional picture under the spectrograms with the sam #
#                  e size. The size will be the same as spectrogram.           #
#                                                                              #
# res:             NEED MORE INFO HERE.                                        #
#                                                                              #
# noise_file:      (optional if noise_type is 'optimal'/ [str,str]) The name o #
#                  f the real data file you want to use as source. The path is #
#                  setted in load_noise function on emily.py. THIS HAS TO BE   #
#                  FIXED LATER!                                                #
#                  The list includes ['date_folder', 'name_of_file']. Date_fol #
#                  der is inside ligo data and contains segments of that day w #
#                  ith the name witch should be the name_of_file.              #
#                                                                              #
# t:               (optinal except psd_mode='default/ float)The duration of th # 
#                  e envelope of every instantiation used to generate the psd. # 
#                  It is used in psd_mode='default'. Prefered to be integral o #
#                  f power of 2. Default is 32.                                #
#                                                                              #
# lags:            (odd int)The number of instantiations we will use for one b #
#                  atch. To make our noise instanstiation indipendent we need  #
#                  a specific number given the detectors. For two detectors gi #
#                  ves n-1 andf or three n(n-1) indipendent instantiations.    #
#                                                                              #
# name:            (optional/string) A special tag you might want to add to yo #
#                  ur saved dataset files. Default is ''.                      #
#                                                                              #
# destination_path:(optional/string) The path where the dataset will be saved, # 
#                  the default is 'null_path+/datasets/'                       #
#                                                                              #
# demo:            (optional/boolean) An option if you want to have a ploted o #
#                  ovrveiw of the data that were generated. It will not work i #
#                  n command line enviroment. Default is false.                #
################################################################################

def data_generator_noise(noise_type         
                         ,length            
                         ,fs                
                         ,size              
                         ,detectors='HLV'   
                         ,spec=False        
                         ,phase=False       
                         ,res=128           
                         ,noise_file=None   
                         ,t=32              
                         ,lags=1
                         ,starting_point=0
                         ,name=''           
                         ,destination_path= null_path+'/datasets/noise/'
                         ,demo=False):      
    
    ## INPUT CHECKING ##########
    #
    
    # noise_type
    if (noise_type!='optimal' and noise_type!='sudo_real' 
        and noise_type!='real'): 
        
        raise ValueError('Wrong type of noise, the only acceptable are:'
                         +' \n\'optimal\'\n\'sudo_real\'\n\'real\'')
    
    #length
    if (not (isinstance(length,float) or isinstance(length,int)) and length>0):
        raise ValueError('The length value has to be a possitive float or'
                         +' integer.')
        
    # fs
    if not isinstance(fs,int) or fs<=0:
        raise ValueError('Sample frequency has to be a positive integer.')
    
    # detectors
    for d in detectors:
        if (d!='H' and d!='L' and d!='V' and d!='K'): 
            raise ValueError('Not acceptable character for detector. '
            +'You should include: \nH for LIGO Hanford\nL for LIGO Livingston'
            +' \nV for Virgo \nK for KAGRA\nFor example: \'HLV\', \'HLVK\'')
    # res
    if not isinstance(res,int):
        raise ValueError('Resolution variable (res) can only be integral')
    
    # noise_file    
    if (noise_type=='sudo_real' or noise_type=='real') and noise_file==None:
        raise TypeError('If you use suno_real or real noise you need a real'
                        +' noise file as a source.')   

    if (noise_file!=None and len(noise_file)==2 
        and isinstance(noise_file[0],str) and isinstance(noise_file[1],str)):
        
        for d in detectors:
            if os.path.isdir( null_path+'/ligo_data/'+str(fs)+'/'+noise_file[0]
                +'/'+d+'/'+noise_file[1]+'.txt'):
                
                raise FileNotFoundError('No such file or directory: \'' 
                    +null_path+'/ligo_data/'+str(fs)+'/'+noise_file[0]
                    +'/'+d+'/'+noise_file[1]+'.txt\'')
                
    # t
    if not isinstance(t,int):
        raise ValueError('t needs to be an integral')
        
    # batch size:
    if not (isinstance(lags, int) and lags%2!=0):
        raise ValueError('lags has to be an odd integer')

    # name
    if not isinstance(name,str): 
        raise ValueError('name optional value has to be a string')
    
    # destination_path
    if not os.path.isdir(destination_path): 
        raise ValueError('No such path '+destination_path)
    #                        
    ########## INPUT CHECKING ## 

    lab={}
    #lab={10:'X', 100:'C', 1000:'M', 10000:'XM',100000:'CM'}  
    # Labels used in saving file
    if size not in lab:
        lab[size]=str(size)    
    
    DATA=[]
    
    ##########################
    #                        #
    # CASE OF OPTIMAL NOISE  #       
    #                        #
    ##########################
    if noise_type=='optimal':
        for i in range(0,size):
            
            if 'H'in detectors:
                
                # Creation of the artificial noise.            
                PSDH,XH,TH=simulateddetectornoise('aligo',t,fs,10,fs/2)
                # Making the noise a TimeSeries
                H_back=TimeSeries(XH,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdH=H_back.asd(1,0.5)                                   
                #Whitening final data                                    
                h=H_back.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)]

            if 'L'in detectors:
                
                # Creation of the artificial noise.            
                PSDL,XL,TL=simulateddetectornoise('aligo',t,fs,10,fs/2)
                # Making the noise a TimeSeries
                L_back=TimeSeries(XL,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdL=L_back.asd(1,0.5)
                #Whitening final data                                           
                l=L_back.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)]

            if 'V'in detectors:
                
                # Creation of the artificial noise.            
                PSDV,XV,TV=simulateddetectornoise('avirgo',t,fs,10,fs/2)
                # Making the noise a TimeSeries
                V_back=TimeSeries(XV,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later  
                asdV=V_back.asd(1,0.5)                                    
                #Whitening final data                                           
                v=V_back.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)]

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))

            row=[dumie,0]  # 'NOISE' = 0 
            DATA.append(row)    
            print('N',i)
                    
            
    ############################
    #                          #
    # CASE OF SUDO-REAL NOISE  #       
    #                          #
    ############################        
    if noise_type =='sudo_real':
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0]
                                                   ,'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0]
                                                   ,'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0]
                                                   ,'V',noise_file[1])
    
        ind=index_combinations(detectors = detectors
                               ,lags = lags
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)
        
        for i in range(0,size):
            
            if 'H'in detectors:
                
                # Calling the real noise segments
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]
                # Generating the PSD of it
                p, f = plt.psd(noiseH, Fs=fs, NFFT=fs, visible=False)    
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDH,XH,TH=simulateddetectornoise([f,p],t,fs,10,fs/2)
                # Making the noise a TimeSeries
                H_back=TimeSeries(XH,sample_rate=fs)  
                # Calculating the ASD so tha we can use it for whitening later
                asdH=H_back.asd(1,0.5)                                    
                #Whitening final data                                           
                h=H_back.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)] 

            if 'L'in detectors:
                
                # Calling the real noise segments
                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]
                # Generating the PSD of it
                p, f = plt.psd(noiseL, Fs=fs, NFFT=fs, visible=False)     
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDL,XL,TL=simulateddetectornoise([f,p],t,fs,10,fs/2)
                # Making the noise a TimeSeries
                L_back=TimeSeries(XL,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later 
                asdL=L_back.asd(1,0.5)                                     
                #Whitening final data
                l=L_back.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)] 

            if 'V'in detectors:
                
                # Calling the real noise segments
                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]  
                # Generating the PSD of it
                p, f = plt.psd(noiseV, Fs=fs, NFFT=fs, visible=False)     
                p, f=p[1::],f[1::]
                # Feeding the PSD to generate the sudo-real noise.            
                PSDV,XV,TV=simulateddetectornoise([f,p],t,fs,10,fs/2)
                # Making the noise a TimeSeries
                V_back=TimeSeries(XV,sample_rate=fs)  
                # Calculating the ASD so tha we can use it for whitening later 
                asdV=V_back.asd(1,0.5)                                    
                #Whitening final data           
                v=V_back.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)] 
            

            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))

            row=[dumie,0]  # 'NOISE' = 0 
            DATA.append(row)    
            print('N',i)
            
            
    #######################
    #                     #
    # CASE OF REAL NOISE  #       
    #                     #
    #######################
    if noise_type =='real':
        
        if 'H' in detectors: noise_segH=load_noise(fs,noise_file[0]
                                                   ,'H',noise_file[1])
        if 'L' in detectors: noise_segL=load_noise(fs,noise_file[0]
                                                   ,'L',noise_file[1])
        if 'V' in detectors: noise_segV=load_noise(fs,noise_file[0]
                                                   ,'V',noise_file[1])
        
        ind=index_combinations(detectors = detectors
                               ,lags = lags
                               ,length = length
                               ,fs = fs
                               ,size = size
                               ,start_from_sec=starting_point)
        
        for i in range(0,size):

            if 'H'in detectors:
                
                # Calling the real noise segments
                noiseH=noise_segH[ind['H'][i]:ind['H'][i]+t*fs]  
                # Making the noise a TimeSeries
                H_back=TimeSeries(noiseH,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later 
                asdH=H_back.asd(1,0.5)                                    
                # Whitening data                                                
                h=H_back.whiten(1,0.5,asd=asdH)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)] 

            if 'L'in detectors:
                # Calling the real noise segments
                noiseL=noise_segL[ind['L'][i]:ind['L'][i]+t*fs]
                # Making the noise a TimeSeries
                L_back=TimeSeries(noiseL,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later
                asdL=L_back.asd(1,0.5)   
                # Whitening data                                                
                l=L_back.whiten(1,0.5,asd=asdL)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)]


            if 'V'in detectors:
                
                # Calling the real noise segments
                noiseV=noise_segV[ind['V'][i]:ind['V'][i]+t*fs]
                # Making the noise a TimeSeries
                V_back=TimeSeries(noiseV,sample_rate=fs)
                # Calculating the ASD so tha we can use it for whitening later   
                asdV=V_back.asd(1,0.5)                                    
                # Whitening data
                v=V_back.whiten(1,0.5,asd=asdV)[int(((t-length)/2)*fs):
                                                int(((t+length)/2)*fs)] 


            dumie=[]
            if 'H' in detectors: dumie.append(np.array(h))
            if 'L' in detectors: dumie.append(np.array(l))
            if 'V' in detectors: dumie.append(np.array(v))

            row=[dumie,0]  # 'NOISE' = 0 
            DATA.append(row)    
            print('N',i , ind['H'][i]/fs)  
                    
            
    # We are gonna use Keras which demands a specific format of datasets
    # So we do that formating here

    data=[]
    labels=[]
    for row in DATA:
        data.append(row[0])
        labels=labels+[row[1]]

    del DATA   
    data, labels= np.array(data), np.array(labels)
    
    
    labels=labels.reshape(1*size,1)
    print(data.shape,labels.shape)


    data = data.transpose((0,2,1))

    print('Shape of data created:   ',data.shape)
    print('Shape of labels created: ',labels.shape)

    if demo==True:
        for i in range(0,10):
            plt.figure(figsize=(10,5))
            
            for j in range(0,len(detectors)):
                plt.plot(np.arange(0,length,1./fs),data[i][:,j]+j*4)
            plt.show()

    
    d={'data': data,'labels': labels}
    save_name=detectors+'_time_'+noise_type+'_noise'+'_'+name+lab[size]
    io.savemat(destination_path+save_name,d)
    print('File '+save_name+'.mat was created')

    
    if spec==True:
        data_spec=[]
        stride ,fft_l = res/fs, res/fs  

        for time_series in data:
            spec=[]
            for i in range(0,len(detectors)):
                # Calculating complex spectrogram
                f, t, spec_gram = spectrogram(time_series[:,i]         
                            , fs 
                            , window='hanning'
                            , nperseg=int(stride*fs)
                            , noverlap=0
                            , nfft =int(fft_l*fs)
                            , mode='complex'
                            ,scaling ='density')
                spectogram=np.array(np.abs(spec_gram))
                # Calculating phase
                if phase==True: phasegram=np.array(np.angle(spec_gram)) 
                spec.append(spectogram)
                if phase==True: spec.append(phasegram)                  

            data_spec.append(spec)

        data_spec=np.array(data_spec)

        data_spec = data_spec.transpose((0,2,3,1))
        #data_spec = np.flip(data_spec,1)    

        print('Shape of data created:   ',data_spec.shape)
        print('Shape of labels created: ',labels.shape)

        d={'data': data_spec,'labels': labels}
        save_name=detectors+'_spec_'+noise_type+'_noise'+'_'+name+lab[size]
        io.savemat(destination_path+save_name,d)
        print('File '+save_name+'.mat was created')

        if demo==True:
            for i in range(0,20):
                if len(detectors)==3:
                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,::2]
                                   /np.max(data_spec[i][:,:,::2])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1::2]
                                   /np.max(data_spec[i][:,:,1::2])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i]
                                   /np.max(data_spec[i])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                elif len(detectors) in [1,2]:

                    if phase==True:
                        plt.figure(figsize=(10,5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_spec[i][:,:,0]
                                   /np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                        plt.subplot(1, 2, 2)
                        plt.imshow(data_spec[i][:,:,1]
                                   /np.max(data_spec[i][:,:,1])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')

                    else:
                        plt.figure(figsize=(10,5))
                        plt.imshow(data_spec[i][:,:,0]
                                   /np.max(data_spec[i][:,:,0])
                                   ,extent=[0,length, int(1/stride) ,int(fs/2)])
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')


            plt.show()
            
            
            

            
################################################################################
#                                                                              #
#  THIS FUNCTION USES THE ABOVE TO CREATE DATASETS READY FOR TRAINING USING    #
#   REAL LIGO DATA YOU ALREADY DOWNLOADED IN A SPECIFIC FORMAT                 #
#                                                                              #
#                                                                              #
# set_type: (list) Must have one of the following formats:                     #
#                                                                              #
# noise --> ['noise'                                                           #
#            ,'real'/'sudo_real'     Only those two options for now            #
#            , detectors(str)        Detectirs of choise                       #
#            , number of sets (int)]                                           #
#            ,(optinoal:) spectogram(bool), phasegram(bool), res(int)]         #
#                                                                              #
#                                                                              #
# cbc --> ['cbc'                                                               #
#            ,'real'/'sudo_real'                                               #
#            , detectors(str)     Detectirs of choise                          #
#            ,'source_cbc'        The source file of injections                #
#            ,[snr1, snr2, ...]                                                #
#         ,(optinoal:) spectogram(bool), phasegram(bool), res(int)]            #
#                                                                              #
# date_ini :(str) The date in witch you want to start the generation in your   #
#           ligo_data file                                                     #
#                                                                              #
# size:   (int)The size of every individual dataset                            #
#                                                                              #
# lags:   (int) The number of lags used in this datasets                       #
#                                                                              #
#  AFTER THIS FINISHES SUCCESFULLY, YOU HAVE TO TIDY THE DATASET USING THE     #
#  NEXT FUNCTION                                                               #
#                                                                              #
################################################################################

def auto_gen(set_type
             ,date_ini
             ,size
             ,fs
             ,length 
             ,lags
             ,t=32
             ,s_name=''):
        
    if len(set_type) not in [4,5,6,7,8]:
        raise TypeError('\'set_type\' must have one of the following'
                +' formats: \n'+'noise --> [\'noise\', \'real\',detectors,'
                +' number of sets (int)] \n'+'cbc --> [\'cbc\',\'real\','
                +' detectors '+',\'cbc_02\',[20, 30, 40 , ...]]')
        
    if set_type[0]=='noise':
        
        
        if set_type[1] in ['sudo_real','real']: 
            noise_type=set_type[1]
        else:
            raise ValueError('noise can be either real or sudo_real')
            
        # detectors
        for d in set_type[2]:
            if (d!='H' and d!='L' and d!='V' and d!='K'): 
                
                raise ValueError('Not acceptable character for detector.'
                    +' You should include: \nH for LIGO Hanford\nL for LIGO'
                    +' Livingston \nV for Virgo \nK for KAGRA\nFor example:'
                    +' \'HLV\', \'HLVK\'')
        
        if (isinstance(set_type[3],int) and set_type[3]>0):
            num_of_sets=set_type[3] 
        else:
            raise ValueError('Number of sets must be integer positive')
            
        spect_b=False
        phase_b=False
        res=128
        if len(set_type)==7:
            spect_b=set_type[4]
            phase_b=set_type[5]
            res=set_type[6]

            
    if set_type[0] in ['cbc','burst']:
        
        if set_type[1] in ['sudo_real','real']: 
            noise_type=set_type[1]
        else:
            raise ValueError('noise can be either real or sudo_real')
            
        # detectors
        for d in set_type[2]:
            if (d!='H' and d!='L' and d!='V' and d!='K'): 
                raise ValueError('Not acceptable character for detector.'
                    +' You should include: \nH for LIGO Hanford\nL for LIGO'
                    +' Livingston \nV for Virgo \nK for KAGRA\nFor example:'
                    +' \'HLV\', \'HLVK\'')
        print(null_path+'/injections/'+set_type[0])

        if set_type[3] in dirlist( null_path+'/injections/'+set_type[0]+'s/'):
            injection_source=set_type[3]
        else:
            raise ValueError('File does not exist')
            
        if len(set_type[4]) > 0:
            for num in set_type[4]:
                if not ((isinstance(num, float) or isinstance(num,int)) 
                        and num >=0 ):
                    
                    raise ValueError('SNR values have to be possitive numbers')
                else:
                    snr_list=set_type[4]
                    num_of_sets=len(snr_list)
        else:
            raise TypeError('SNR must be specified as a list [20, 15 , ]')
         
        spect_b=False
        phase_b=False
        res=128
        if len(set_type)==8:
            spect_b=set_type[5]
            phase_b=set_type[6]
            res=set_type[8]
    # Calculation of how many segments we need for the given requirements. 
    #There is a starting date and if it needs more it goes to the next one.

    
    date_list=dirlist( null_path+'/ligo_data/2048')

    # Generating a list with all the available dates
    date=date_list[date_list.index(date_ini)]
    # Calculating the index of the initial date
    counter=date_list.index(date_ini)             

    
    # Calculation of the duration 
    # Here we infere the duration needed given the lags used in the method

    if lags==1:
        duration_need = size*num_of_sets*length
        tail_crop=0
    if lags%2 == 0:
        duration_need = ceil(size*num_of_sets/(lags*(lags-2)))*lags*length
        tail_crop=lags*length
    if lags%2 != 0 and lags !=1 :
        duration_need = ceil(size*num_of_sets/(lags*(lags-1)))*lags*length
        tail_crop=lags*length


    # Creation of lists that indicate characteristics of the segments based on the duration needed. 
    # These are used for the next step.

    duration_total = 0
    duration, gps_time, seg_list=[],[],[]

    while duration_need > duration_total:
        counter+=1
        segments=dirlist( null_path+'/ligo_data/'+str(fs)+'/'+date+'/H')
        print(date)
        for seg in segments:
            for j in range(len(seg)):
                if seg[j]=='_': 
                    gps, dur = seg[j+1:-5].split('_')
                    break

            duration.append(int(dur))
            gps_time.append(int(gps))
            seg_list.append([date,seg])

            duration_total+=(int(dur)-3*t-tail_crop)
            print('    '+seg)

            if duration_total > duration_need: break

        if counter==len(date_list): counter==0        
        date=date_list[counter]

    # Generation of lists with the parameters that the dataset_generator_noice
    # the automated datasets.

    size_list=[]            # Sizes for each generation of noise 
    starting_point_list=[]  # Starting points for each generation of noise(s)
    seg_list_2=[]           # Segment names for each generation of noise
    number_of_set=[]        # No of set that this generation of noise will go
    name_list=[]            # List with the name of the set to be generated
    number_of_set_counter=0 # Counter that keeps record of how many                                         # instantiations have left to be generated 
                            # to complete a set
    
    
    if set_type[0]=='noise':
        set_num=1

        for i in range(len(np.array(seg_list)[:,1])):

            # local size indicates the size of the file left for generation of  
            # datasets, when it is depleted the algorithm moves to the next 
            # segment. Here we infere the local size given the lags used in the             # method

            if lags==1:    # zero lag case
                local_size=ceil((duration[i]-3*t-tail_crop)/length)
            if lags%2 == 0:
                local_size=ceil((duration[i]-3*t-tail_crop
                                )/length/lags)*lags*(lags-2)
            if lags%2 != 0 and lags !=1 :
                local_size=ceil((duration[i]-3*t-tail_crop
                                )/length/lags)*lags*(lags-1)

            # starting point always begins with the window of the psd 
            # to avoid deformed data of the begining    
            local_starting_point=t

            # There are three cases when a segment is used.
            # 1. That it has to fill a previous set first and then 
            # move to the next
            # 2. That it is the first set so there is no previous set to fill
            # 3. It is the too small to fill so its only part of a set.
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
                if lags==1: 
                    local_starting_point+=(size-number_of_set_counter)*length
                    
                if lags%2 == 0:
                    local_starting_point+=(ceil((size-number_of_set_counter)
                    /lags/(lags-2))*lags*length)
                    
                if lags%2 != 0 and lags !=1 :
                    local_starting_point+=(ceil((size-number_of_set_counter)
                    /lags/(lags-1))*lags*length)
                
                number_of_set_counter+=(size-number_of_set_counter)

                # If this generation completes the size of a whole set 
                # (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(set_num)
                    if size_list[-1]==size: 
                        name_list.append(str(set_num))
                    else:
                        name_list.append('part_of_'+str(set_num))
                    set_num+=1
                    number_of_set_counter=0
                    if set_num > num_of_sets: break

                elif number_of_set_counter < size:
                    number_of_set.append(set_num)
                    name_list.append('part_of_'+str(set_num))
                    
            if (len(size_list) == 0 or number_of_set_counter==0):
                while local_size >= size:
                    # Generate data with size 10000 with final name of       
                    # 'name_counter'
                    size_list.append(size)
                    seg_list_2.append(seg_list[i])
                    starting_point_list.append(local_starting_point)

                    #Update the the values for the next set
                    local_size-=size
                    if lags==1: local_starting_point+=size*length
                    if lags%2 == 0:
                        local_starting_point+=(ceil(size/lags/(lags-2))
                                               *lags*length)
                    if lags%2 != 0 and lags !=1 : 
                        local_starting_point+=(ceil(size/lags
                        /(lags-1))*lags*length)
                        
                    number_of_set_counter+=size

                    # If this generation completes the size of a whole set
                    # (with size=size) it changes the labels
                    if number_of_set_counter == size:
                        number_of_set.append(set_num)
                        if size_list[-1]==size: 
                            name_list.append(str(set_num))
                        else:
                            name_list.append('part_of_'+str(set_num))
                        set_num+=1
                        if set_num >= num_of_sets: break
                        number_of_set_counter=0

            if (local_size < size and local_size >0 and set_num < num_of_sets):
                
                # Generate data with size 'local_size' with local name 
                # to be fused with later one
                size_list.append(local_size)
                seg_list_2.append(seg_list[i])
                starting_point_list.append(local_starting_point)

                # Update the the values for the next set
                
                # Saving a value for what is left for the next seg to generate
                number_of_set_counter+=local_size  

                # If this generation completes the size of a whole set 
                # (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(set_num)
                    if size_list[-1]==size: 
                        name_list.append(str(set_num))
                    else:
                        name_list.append('part_of_'+str(set_num))
                    set_num+=1
                    if set_num > num_of_sets: break
                    number_of_set_counter=0

                elif number_of_set_counter < size:
                    number_of_set.append(set_num)
                    name_list.append('part_of_'+str(set_num))

    
    if set_type[0] in ['cbc','burst']:
        
        set_num=0

        for i in range(len(np.array(seg_list)[:,1])):
                

            # local size indicates the size of the file left for generation 
            # of datasets, when it is depleted the algorithm moves to the next               # segment. Here we infere the local size given the lags used in 
            # the method.

            if lags==1:    # zero lag case
                local_size=ceil((duration[i]-3*t-tail_crop)/length)
            if lags%2 == 0:
                local_size=ceil((duration[i]-3*t-tail_crop)
                                /length/lags)*lags*(lags-2)
            if lags%2 != 0 and lags !=1 :
                local_size=ceil((duration[i]-3*t-tail_crop)
                                /length/lags)*lags*(lags-1)
                

            # starting point always begins with the window of the psd to avoid
            # deformed data of the begining    
            local_starting_point=t

            # There are three cases when a segment is used.
            # 1. That it has to fill a previous set first and then move 
            # to the next
            # 2. That it is the first set so there is no previous set to fill
            # 3. It is the too small to fill so its only part of a set.
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
                if lags==1:
                    local_starting_point+=((size-number_of_set_counter)
                                           *length)
                if lags%2 == 0:
                    local_starting_point+=(ceil((size
                    -number_of_set_counter)/lags/(lags-2))*lags*length)
                if lags%2 != 0 and lags !=1 :
                    local_starting_point+=(ceil((size
                    -number_of_set_counter)/lags/(lags-1))*lags*length)
                    
                number_of_set_counter += (size-number_of_set_counter)

                # If this generation completes the size of a whole set
                # (with size=size) it changes the labels
                if number_of_set_counter == size:
                    number_of_set.append(snr_list[set_num])
                    if size_list[-1]==size: 
                        name_list.append(str(snr_list[set_num])+s_name)
                    else:
                        name_list.append('part_of_'
                            +str(snr_list[set_num])+s_name)
                    set_num+=1
                    number_of_set_counter=0
                    if set_num >= num_of_sets: break
                        
                elif number_of_set_counter < size:
                    number_of_set.append(snr_list[set_num])
                    name_list.append('part_of_'+str(snr_list[set_num])+s_name)

            if (len(size_list) == 0 or number_of_set_counter==0):
                while local_size >= size:
                    # Generate data with size 10000 with final name of 
                    # 'name_counter'
                    size_list.append(size)
                    seg_list_2.append(seg_list[i])
                    starting_point_list.append(local_starting_point)

                    #Update the the values for the next set
                    local_size-=size
                    if lags==1: local_starting_point+=size*length
                    if lags%2 == 0: 
                        local_starting_point+=(ceil(size/lags
                                                    /(lags-2))*lags*length)
                    if lags%2 != 0 and lags !=1 :
                        local_starting_point+=(ceil(size/lags
                                                    /(lags-1))*lags*length)
                    number_of_set_counter+=size

                    # If this generation completes the size of a whole set
                    # (with size=size) it changes the labels
                    if number_of_set_counter == size:
                        number_of_set.append(snr_list[set_num])
                        if size_list[-1]==size: 
                            name_list.append(str(snr_list[set_num])+s_name)
                        else:
                            name_list.append('part_of_'
                                             +str(snr_list[set_num])+s_name)
                        set_num+=1
                        if set_num >= num_of_sets: break
                        number_of_set_counter=0

            if (local_size < size and local_size >0 and set_num < num_of_sets):
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
                        name_list.append(str(snr_list[set_num])+s_name)
                    else:
                        name_list.append('part_of_'
                                         +str(snr_list[set_num])+s_name)
                    set_num+=1
                    if set_num >= num_of_sets: break
                    number_of_set_counter=0

                elif number_of_set_counter < size:
                    number_of_set.append(snr_list[set_num])
                    name_list.append('part_of_'+str(snr_list[set_num])+s_name)


    d={'segment' : seg_list_2, 'size' : size_list 
       , 'start_point' : starting_point_list, 'set' : number_of_set
       , 'name' : name_list}

    print('These are the details of the datasets to be generated: \n')
    #print((d['segment']), (d['size']),(d['start_point']) ,(d['name']))
    for i in range(len(d['segment'])):
        print(d['segment'][i], d['size'][i], d['start_point'][i] ,d['name'][i])
        
    print('Should we proceed to the generation of the following'
          +' data y/n ? \n \n')

    answer=input()
    
    if answer in ['no','n']:
        print('Exiting procedure')
        return
        
    elif answer in ['yes','y']:
        
        print('Of course lord commander ...')
        print('Type the name of the dataset directory:')
        
        dir_name=input()
        
        if set_type[0]=='noise':
            path= null_path+'/datasets/noise/'+set_type[1]+'/'
        elif set_type[0] in ['cbc','burst']:
            path= null_path+'/datasets/'+set_type[0]+'/'

        print('The current path of the directory is: \n'+path+'\n')
        print('Do you wanna change the path y/n ?')
        
        answer2=input()

        if answer2 in ['yes','y']:
            print('Insert complete new path ex. /home/....')
            path=input()
            if path=='exit': return
            
            while not (os.path.isdir(path)):
                print('Path: '+path+' is not existing. Try again or type '
                      +'exit to exit the procedure.')
                path=input()
                if path=='exit': return
            
        if answer2 in ['no','n']:
            if os.path.isdir(path+dir_name):
                print('Already existing '+dir_name+' directory, do want to'
                      +' delete it? y/n')
                answer3=input()
                if answer3=='y':
                    os.system('rm -r '+path+dir_name)
                elif answer3=='n':
                    return
            print('Initiating procedure ...')
            os.system('mkdir '+path+dir_name)
            print('Creation of directory complete: '+path+dir_name)
            
        os.system('cd '+path+dir_name)
                

        for i in range(len(d['segment'])):

            with open(path+dir_name+'/'+'gen_'+d['name'][i]+'_'
                +str(d['size'][i])+'.py','w') as f:
                
                f.write('import sys \n')
                
                f.write('sys.path.append(\'/home/vasileios.skliris/mly/\')\n')# # # #

                f.write('sys.path.append(\''+null_path+'/\')\n')
                f.write('from mly.generators import data_generator_inj'+
                        ', data_generator_noise \n')
                
                if set_type[0] in ['cbc','burst']:

                    comand=(
                    'data_generator_inj(parameters=[\''
                     +set_type[0]+'s/'+injection_source+'\',\''+noise_type+'\','
                     +str(d['set'][i])+']'+
                               ',length='+str(length)+           
                               ',fs='+str(fs)+              
                               ',size='+str(d['size'][i])+             
                               ',detectors=\''+set_type[2]+'\''+
                               ',spec='+str(spect_b)+
                               ',phase='+str(phase_b)+
                               ',res='+str(res)+
                               ',noise_file='+str(d['segment'][i])+
                               ',t='+str(t)+             
                               ',lags='+str(lags)+
                               ',starting_point='+str(d['start_point'][i])+
                               ',name=\''+str(d['name'][i])+'_\''+
                               ',destination_path=\''+path+dir_name+'/\''+
                               ',demo=False)')
                    
                elif set_type[0]=='noise':
                    
                    comand=(
                    'data_generator_noise(noise_type=\''+noise_type+'\''+
                               ',length='+str(length)+           
                               ',fs='+str(fs)+              
                               ',size='+str(d['size'][i])+             
                               ',detectors=\''+set_type[2]+'\''+
                               ',spec='+str(spect_b)+
                               ',phase='+str(phase_b)+
                               ',res='+str(res)+
                               ',noise_file='+str(d['segment'][i])+
                               ',t='+str(t)+             
                               ',lags='+str(lags)+
                               ',starting_point='+str(d['start_point'][i])+
                               ',name=\''+str(d['name'][i])+'_\''+
                               ',destination_path=\''+path+dir_name+'/\''+
                               ',demo=False)')
                
                f.write(comand+'\n')

        with open(path+dir_name+'/'+'auto_gen.sh','w') as f2:
            
            f2.write('#!/bin/sh \n\n')
            for i in range(len(d['segment'])):
                
                f2.write('nohup python '+path+dir_name+'/'+'gen_'+d['name'][i]
                    +'_'+str(d['size'][i])+'.py > ' +path+dir_name+'/out_'
                    +d['name'][i]+'_'+str(d['size'][i])+'.out & \n' )
                
        
        with open(path+dir_name+'/info.txt','w') as f3:
            f3.write('INFO ABOUT DATASETS GENERATION \n\n')
            f3.write('fs: '+str(fs)+'\n')
            f3.write('length: '+str(length)+'\n')
            f3.write('window: '+str(t)+'\n')
            f3.write('lags: '+str(lags)+'\n'+'\n')
            
            for i in range(len(d['segment'])):
                f3.write(d['segment'][i][0]+' '+d['segment'][i][1]
                         +' '+str(d['size'][i])+' '
                         +str(d['start_point'][i])+'_'+d['name'][i]+'\n')
            
        print('All set. Initiate dataset generation y/n?')
        answer4=input()
        
        if answer4 in ['y','Y']:
            os.system('sh '+path+dir_name+'/auto_gen.sh')
            return
        else:
            print('Data generation canceled')
            os.system('cd')
            os.system('rm -r '+path+dir_name)
            return
                
    return


################################################################################
#                                                                              #
#  THIS FUNCTIONS CHECKS THAT auto_gen WORKED SUCCESFULLY AND THEN MERGES ALL  #
#  THE DATASETS THAT BELONG TO THE SAME GROUP. IT ALSO DELETES ALL THE FILES   #
#  auto_gen MADE THAT ARE NOT NEEDED ANY MORE.                                 #
#  path: (str) The path to the file that needs tidying.                        #
#                                                                              #
#  final_size: (int) The final size of the datasets that need tidying. The     #
#              auto_gen function created groups based on that size.            #
#                                                                              #
################################################################################

def finalise_gen(path,final_size=10000):

    files=dirlist(path)
    merging_flag=False

    print('Running diagnostics for file: '+path+'  ... \n') 
    py, dat=[],[]
    for file in files:
        if file[-3:]=='.py':
            py.append(file)
        if file[-4:]=='.mat': #In the future the type of file might change
            dat.append(file)

    # Checking if all files that should have been generated 
    # from auto_gen are here
    
    if len(dat)==len(py):
        print('Files succesfully generated, all files are here')
        print(len(dat),' out of ',len(py))
        merging_flag=True  # Declaring that merging can happen now
    
    elif len(dat)>len(py):
        print('There are some files already merged or finalised\n')
        print('Files succesfully generated, all files are here')
        print(len(dat),' out of ',len(py))
        merging_flag=True  # Declaring that merging can happen now

    
    else:
        failed_py=[]
        for i in range(len(py)):
            py_id=py[i][4:-3]
            counter=0
            for dataset in dat:
                if py_id in dataset:
                    counter=1
            if counter==0:
                print(py[i],' failed to proceed')
                failed_py.append(py[i])



    if merging_flag==False:
        print('\n\nDo you want to try and run again the failed procedurs? y/n')
        answer1=input()
        
        if answer1 in ['Y','y','yes']:
            with open(path+'/auto_gen_redo.sh','w') as fnew:
                fnew.write('#!/bin/sh\n\n')
                with open(path+'/auto_gen.sh','r') as f:
                    for line in f:
                        for pyid in failed_py:
                            if pyid in line:
                                fnew.write(line+'\n')
                                break

            print('All set. The following generators are going to run again:\n')
            for pyid in failed_py:
                print(pyid)

            print('\n\nInitiate dataset generation y/n?')
            answer2=input()

            if answer2 in ['y','Y','yes']:
                os.system('sh '+path+'/auto_gen_redo.sh')
                return
            else:
                print('Data generation canceled')
                os.system('rm '+path+'/auto_gen_redo.sh')
                return
            
            
        elif answer1 in ['N','n','no','exit']:
            print('Exiting procedure')
            return
        
        
    if merging_flag==True:
            
        print('\n Do you wanna proceed to the merging of the datasets? y/n')
        answer3=input()
              
        if answer3 in ['n','N','no','NO','No']:
            print('Exiting procedure')
            return
        
        elif answer3 in ['y','Y','yes','Yes']:  
            
            # Labels used in saving file
            lab={10:'X', 100:'C', 1000:'M', 10000:'XM',100000:'CM'}  
            if final_size not in lab:
                lab[final_size]=str(final_size)
            
            # Geting the details of files from their names
            set_name,set_id, set_size,ids,new_dat=[],[],[],[],[]
            for dataset in dat:
                if 'part_of' not in dataset:
                    new=dataset.split('_')
                    
                    if 'SNR' in dataset:
                        new.pop(-1)
                        new.pop(-1)

                        final_name=('_'.join(new)+'_'+lab[final_size]+'.mat')
                    elif 'noise' in dataset.split('_')[-3]:
                        new.pop(-1)
                        new[-1]='No'+new[-1]
                        final_name=('_'.join(new)+'_'+lab[final_size]+'.mat')

                    os.system('mv '+path+dataset+' '+path+final_name)

                else:
                    new_dat.append(dataset)
                    new=dataset.split('part_of_')
                    set_name.append(dataset)
                    set_id.append(new[1].split('_')[0])
                    set_size.append(new[1].split('_')[1].split('.')[0])
                    if new[1].split('_')[0] not in ids:
                        ids.append(new[1].split('_')[0])
                    
                    
            dat=new_dat

            # Creating the inputs for the function data_fusion
            merging_index=[]
            for ID in ids:
                sets,sizes=[],[]
                for i in range(len(dat)):
                    
                    if set_id[i]==ID:
                        sets.append(dat[i])
                        sizes.append(set_size[i])
                merging_index.append([ID,sets,sizes])


            # Initiate merging of files
            for k in range(len(merging_index)):
                
                if 'SNR' in merging_index[k][1][0].split('part_of_')[0][-6:]:
                    final_name=(merging_index[k][1][0].split('part_of_')[0]
                                +lab[final_size]+'.mat')
                if 'noise' in merging_index[k][1][0].split('part_of_')[0][-6:]:
                    final_name=(merging_index[k][1][0].split('part_of_')[0]+'No'
                                +ids[k]+'_'+lab[final_size]+'.mat')



                data_fusion(names=merging_index[k][1]
                    ,sizes=None # Given the sum is right
                    ,save=path+final_name
                    ,data_source_file=path)    

            print('\n\nDataset is complete!')
             
                
            # Deleting unnescesary file in the folder
            print('\n Do you want to delete unnecessary files? y/n')
            
            answer4=input()
            if answer4 in ['n','N','no','NO','No']:
                return

            elif answer4 in ['y','Y','yes','Yes']: 

                for file in dirlist(path):
                    
                    if ('info' not in file) and (lab[final_size] not in file):

                        os.system('rm '+path+file)
            
            
            print('\n Do you wanna check the datasets for possibe nan values? y/n')
            answer5=input()

            if answer3 in ['y','Y','yes','Yes']:
                files=dirlist(path)

                nancount=0
                nancount_set=0

                for file in files:
                    if file[-4:]=='.mat': #In the future the type of file might change
                        data = io.loadmat(path+file)
                        X = data['data']
                        Y = data['labels']
                        for i in range(len(X)):
                            nan_alert=False
                            for det in [0,1,2]:
                                for num in X[i][:,det]:
                                    if isnan(num):
                                        nancount+=1
                                        nan_alert=True
                            if nan_alert==True:
                                X[i]=X[np.random.randint(0,i)]
                                nancount_set+=1
                        print(str(nancount)+' elements found in '+str(nancount_set)+' sets'
                              +' and now they are replased with other sets')
                        d={'data': X,'labels': Y}
                        io.savemat(path+file,d)
                
                print('File is finalised')
                return



    



            
    
                    
        
        

        












    



            
    
                    
        
        













            
    
                    
        
        










        

        



    
        

        
        
    
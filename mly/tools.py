import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil 
import os


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

def dirlist(filename):                         
    fn=os.listdir(filename)      
    fn_clean=[]
    for i in range(0,len(fn)): 
        if fn[i][0]!='.':
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
    

    
def toCategorical(labels,translation = True):
    classes=[]
    classesNumerical=[]
    translationDict={}
    num=0
    for label in labels:
        if label not in classes: 
            classes.append(label)
            classesNumerical.append(num)
            num+=1

    classesCategorical = []
    for i in range(len(classes)):
        categorical = len(classes)*[0]
        categorical[i]=1
        classesCategorical.append(categorical)
        print(classes[i])
        translationDict[classes[i]] = categorical
    labelsCategorical=[]
    for label in labels:
        labelsCategorical.append(translationDict[label])
    labelsCategorical=np.asarray(labelsCategorical)
    if translation==True:
        return(labelsCategorical,translationDict)
    else:
        return(labelsCategorical)
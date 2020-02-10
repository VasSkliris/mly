import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import pickle
from math import ceil

from keras import optimizers, initializers
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

# Keras layers
from keras.models import Sequential, load_model, Model
from keras.layers import Dense,Conv1D, Conv2D, MaxPool2D, MaxPool1D, Flatten, concatenate, Input ,Activation
from keras.layers import Dropout, BatchNormalization

import os
from random import shuffle

from .__init__ import *

null_path=nullpath()
0
################################################################    
## This function trains a model with the given compiled model ##
################################################################
 
def train_model(model               # Model or already saved model from directory
               ,dataset             # Dataset to train with
               ,epoch               # Epochs of training
               ,batch               # Batch size of the training 
               ,split               # Split ratio of TEST / TRAINING data
               ,classes=2           # Number of classes used in this training
               ,save_model=False  # (optional) I you want to save the model, assign name
               ,data_source_path= null_path+'/datasets/'
               ,model_source_path= null_path+'/trainings/'):

    if isinstance(model,str): 
        model_fin=load_model(model_source_path+model+'.h5')
    else:
        model_fin=model
        
    if isinstance(dataset,str):
        data = io.loadmat(data_source_path+dataset+'.mat')
        X = data['data']
        Y = data['labels']

        Y = np_utils.to_categorical(Y, classes) # [0] --> [1 0] = NOISE if classes=2
                                                # [1] --> [0 1] = SIGNAL
        print('... Loading file '+dataset+' with data shape:  ',X.shape)
    
    else:
        X, Y = dataset[0], dataset[1]
        Y = np_utils.to_categorical(Y, classes)
        print('... Loading dataset with data shape:  ',X.shape)

    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=split,random_state=0)
    print('... Spliting the dada with ratio: ',split)
    print('... and final shapes of: ',X_train.shape, X_test.shape)
    

    hist=model_fin.fit(X_train
                   ,Y_train
                   ,epochs=epoch
                   ,batch_size=batch
                   ,validation_data=(X_test, Y_test))
    if isinstance(save_model,str):
        model_fin.save(save_model+'.h5')
    return(hist.history)





########################################################
## A function to import a model and test in selected  ##
## dataset. Just use name of dataset. Modify the path ##
## for personal use.                                  ##
########################################################

def test_model(model              # Model to test
               ,test_data         # Testing dateset
               ,extended=False    # Printing extra stuff   NEEDS DEBUGIN
               ,classes = 2       # Number of classes
               ,data_source_path= null_path+'/datasets/'
               ,model_source_path= null_path+'/trainings/'):
    
    if isinstance(model,str):
        trained_model = load_model(model_source_path+ model +'.h5')
    else:
        trained_model = model    #If model is not already in the script you import it my calling the name
    
    if isinstance(test_data,str):
        data = io.loadmat(data_source_path+test_data+'.mat')
        X = data['data']
        Y = data['labels']
        
        #Y_categ = np_utils.to_categorical(Y, 2) # [0] --> [1 0] = NOISE
                                                 # [1] --> [0 1] = SIGNAL       
    else:
        X, Y = test_data[0], test_data[1]
        Y = np_utils.to_categorical(Y, classes)
        print('... Loading dataset with data shape:  ',X.shape)
        

    
    # Confidence that the data is [NOISE | SIGNAL] = [0 1] = [[1 0],[0 1]]
    predictions=trained_model.predict_proba(X, batch_size=1, verbose=1)
    
    #scores=predictions[predictions[:,0].argsort()] 
    
    if extended==True:
        pr1, pr0 = [], []
        true_labels=[]

        for i in range(0,len(Y)):
            if Y[i][1]==1:
                true_labels.append('SIGNAL')
                pr1.append(predictions[i])

            elif Y[i][1]==0:
                true_labels.append('NOISE')
                pr0.append(predictions[i])
        
                       
        pr0, pr1 = np.array(pr0), np.array(pr1)

        plt.ioff()
        plt.figure(figsize=(15,10))
        if len(pr0)!=0:
            n0, b0, p0 = plt.hist(pr0[:,1],bins=1000,log=True,cumulative=-1,color='r',alpha=0.5)
        if len(pr1)!=0:
            n1, b1, p1 = plt.hist(pr1[:,1],bins=1000,log=True,cumulative=1,color='b',alpha=0.5)
        #plt.ylabel('Counts')
        #plt.title('Histogram of '+test_data)
        
        new_name=[]
        for i in test_data[::-1]:
            if i!='/':
                new_name.append(i)
            else:
                break  
        new_name=''.join(new_name[::-1])
        fig_name=new_name+'.png'
        plt.savefig(fig_name)
    
    return(predictions)

##################################################################################################################################################################################################################################################################

def lr_change(model,new_lr):
    old_lr=K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr,new_lr)
    lr=K.get_value(model.optimizer.lr)

    print('Learning rate changed from '+str(old_lr)+' to '+str(lr))
    
    return

##################################################################################################################################################################################################################################################################

def save_history(histories,name='nonamed_history',save=False,extendend=True): 

    train_loss_history = []
    train_acc_history=[]
    val_loss_history = []
    val_acc_history=[]

    for new_history in histories:
        val_loss_history = val_loss_history + new_history['val_loss']
        val_acc_history = val_acc_history + new_history['val_acc'] 
        train_loss_history = train_loss_history + new_history['loss'] 
        train_acc_history = train_acc_history + new_history['acc'] 

    history_total={'val_acc': val_acc_history
                   , 'acc': train_acc_history
                   , 'val_loss':val_loss_history
                   , 'loss': train_loss_history,}
    if save == True:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(history_total, f, pickle.HIGHEST_PROTOCOL)
    
    if extendend == True:
        epochs=np.arange(1,len(history_total['val_acc'])+1)
        
        plt.ioff()
        plt.figure(figsize=(15,10))
        plt.plot(epochs,history_total['acc'],'b')
        plt.plot(epochs,history_total['val_acc'],'r')
        plt.title('Accuracy of validation and testing')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
         
        new_name=[]                      # 
        for i in name[::-1]:             #
            if i!='/':
                new_name.append(i)       # routine to avoid naming errors with paths
            else:
                break                    #
        new_name=''.join(new_name[::-1]) # 
        fig_name=new_name+'.png'         #
        plt.savefig(fig_name)

    return(history_total)

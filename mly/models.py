from keras import optimizers, initializers
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

# Keras layers
from keras.models import Sequential, load_model, Model
from keras.layers import Dense,Conv1D, Conv2D, MaxPool2D, MaxPool1D, Flatten, concatenate, Input ,Activation
from keras.layers import Dropout, BatchNormalization    

from .__init__ import *


def conv_model_1D(parameter_matrix, INPUT_SHAPE, LR, verbose=True):
    
    ####################################################
    ##  This type of format for the input data is     ##
    ##  suggested to avoid errors:                    ##
    ##                                                ##
    ##  CORE =     ['C','C','C','C','F','D','D','DR'] ##
    ##  MAX_POOL = [ 2,  0,  0,  2,  0,  0,  0,  0  ] ##
    ##  FILTERS =  [ 8, 16, 32, 64,  0, 64, 32,  0.3] ##
    ##  K_SIZE =   [ 3,  3,  3,  3,  0,  0,  0,  0  ] ## 
    ##                                                ##
    ##  PM=[CORE,MAX_POOL,FILTERS,K_SIZE]             ## 
    ##  in_shape = (52,80,3)                          ##
    ##  lr = 0.00002                                  ##
    ####################################################
    
    CORE = parameter_matrix[0]
    MAX_POOL = parameter_matrix[1]
    FILTERS = parameter_matrix[2]
    K_SIZE = parameter_matrix[3]

    
    model=Sequential()
    receipt=[]

    inis=initializers.he_normal(seed=None)

    for i in range(0,len(CORE)):

        if CORE[i]=='C' and i==0:
            model.add(Conv1D(filters=FILTERS[i]
                             , kernel_size=K_SIZE[i]
                             , activation='elu'
                             , input_shape=INPUT_SHAPE
                             , kernel_initializer=inis,bias_initializer=initializers.Zeros()))
            model.add(BatchNormalization())

            receipt.append('INPUT <------ SHAPE: '+str(INPUT_SHAPE))
            receipt.append('CONV 1D --> FILTERS: %3d KERNEL SIZE: %2d ' % (FILTERS[i], K_SIZE[i])  )
            receipt.append('Bach Normalization')

        if CORE[i]=='C' and i!=0:
            model.add(Conv1D(filters=FILTERS[i], kernel_size=K_SIZE[i], activation='elu'))
            model.add(BatchNormalization())
            receipt.append('CONV 1D --> FILTERS: %3d KERNEL SIZE: %2d ' % (FILTERS[i], K_SIZE[i])  )
            receipt.append('Bach Normalization')


        if MAX_POOL[i]!=0 :
            model.add(MaxPool1D(pool_size=MAX_POOL[i], strides=MAX_POOL[i]))
            model.add(BatchNormalization())
            receipt.append('MAX POOL 1D --> KERNEL SHAPE: %d STRIDE: %d ' % (MAX_POOL[i], MAX_POOL[i]))
            receipt.append('Bach Normalization')

        if CORE[i]=='F': 
            model.add(Flatten())
            receipt.append('<---- FLATTEN ---->')

        if CORE[i]=='D':
            model.add(Dense(FILTERS[i], activation='elu'))
            model.add(BatchNormalization())
            receipt.append('DENSE --> FILTERS: %3d' % FILTERS[i])
            receipt.append('Bach Normalization')


        if CORE[i]=='DR': 
            model.add(Dropout(FILTERS[i]))
            receipt.append('DROP OUT --> '+str(int(100*FILTERS[i]))+'  % ' )

    model.add(Dense(2, activation='softmax'))
    receipt.append('OUTPUT --> SOFTMAX 2 CLASSES')

    opt=optimizers.Nadam(lr=LR
                         , beta_1=0.9
                         , beta_2=0.999
                         , epsilon=1e-8
                         , schedule_decay=0.000002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if verbose==True:
        for line in receipt:
            print(line)

    return(model)


def conv_model_2D(parameter_matrix, INPUT_SHAPE, LR, verbose=True):
    
    ####################################################
    ##  This type of format for the input data is     ##
    ##  suggested to avoid errors:                    ##
    ##                                                ##
    ##  CORE =     ['C','C','C','C','F','D','D','DR'] ##
    ##  MAX_POOL = [ 2,  0,  0,  2,  0,  0,  0,  0  ] ##
    ##  FILTERS =  [ 8, 16, 32, 64,  0, 64, 32,  0.3] ##
    ##  K_SIZE =   [ 3,  3,  3,  3,  0,  0,  0,  0  ] ## 
    ##                                                ##
    ##  PM=[CORE,MAX_POOL,FILTERS,K_SIZE]             ## 
    ##  in_shape = (52,80,3)                          ##
    ##  lr = 0.00002                                  ##
    ####################################################
    
    
    CORE = parameter_matrix[0]
    MAX_POOL = parameter_matrix[1]
    FILTERS = parameter_matrix[2]
    K_SIZE = parameter_matrix[3]

    
    model=Sequential()
    receipt=[]

    inis=initializers.he_normal(seed=None)

    for i in range(0,len(CORE)):

        if CORE[i]=='C' and i==0:
            model.add(Conv2D(filters=FILTERS[i]
                             , kernel_size=K_SIZE[i]
                             , activation='elu'
                             , input_shape=INPUT_SHAPE
                             , kernel_initializer=inis,bias_initializer=initializers.Zeros()))
            model.add(BatchNormalization())

            receipt.append('INPUT <------ SHAPE: '+str(INPUT_SHAPE))
            receipt.append('CONV 2D --> FILTERS: %3d KERNEL SIZE: %2d ' % (FILTERS[i], K_SIZE[i])  )
            receipt.append('Bach Normalization')

        if CORE[i]=='C' and i!=0:
            model.add(Conv2D(filters=FILTERS[i], kernel_size=K_SIZE[i], activation='elu'))
            model.add(BatchNormalization())
            receipt.append('CONV 2D --> FILTERS: %3d KERNEL SIZE: %2d ' % (FILTERS[i], K_SIZE[i])  )
            receipt.append('Bach Normalization')


        if MAX_POOL[i]!=0 :
            model.add(MaxPool2D(pool_size=(MAX_POOL[i],MAX_POOL[i]), strides=MAX_POOL[i]))
            model.add(BatchNormalization())
            receipt.append('MAX POOL 2D --> KERNEL SHAPE:[%d X %d] STRIDE: %d ' % (MAX_POOL[i], MAX_POOL[i], MAX_POOL[i]))
            receipt.append('Bach Normalization')

        if CORE[i]=='F': 
            model.add(Flatten())
            receipt.append('<---- FLATTEN ---->')

        if CORE[i]=='D':
            model.add(Dense(FILTERS[i], activation='elu'))
            model.add(BatchNormalization())
            receipt.append('DENSE --> FILTERS: %3d' % FILTERS[i])
            receipt.append('Bach Normalization')


        if CORE[i]=='DR': 
            model.add(Dropout(FILTERS[i]))
            receipt.append('DROP OUT --> '+str(int(100*FILTERS[i]))+' % ' )

    model.add(Dense(2, activation='softmax'))
    receipt.append('OUTPUT --> SOFTMAX 2 CLASSES')

    opt=optimizers.Nadam(lr=LR
                         , beta_1=0.9
                         , beta_2=0.999
                         , epsilon=1e-8
                         , schedule_decay=0.000002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if verbose==True:
        for line in receipt:
            print(line)

    return(model)
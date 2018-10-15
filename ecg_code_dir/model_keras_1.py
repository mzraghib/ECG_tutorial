#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:58:00 2018

@author: zuhayr
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import BatchNormalization, MaxPool2D, Conv2D, Activation, Dense, Flatten, Dropout, InputLayer
from tensorflow.python.keras import optimizers
from sklearn.metrics import recall_score

#import tensorflow.python.keras.backend as K
import numpy as np


from tensorflow.python.keras.utils import multi_gpu_model
   
def model_from_paper(multi_gpu = True, num_gpus = 4):
    """
    
    Defines hyperparameters and compiles the CNN model used in the paper:
        https://arxiv.org/abs/1804.06812

    Returns
    -------
    A Keras sequential model object

    """  

    model = Sequential()
#    model.add(InputLayer(input_shape=(128,128,1)))    
    model.add(Conv2D(64, (3,3),
                     strides = (1,1), 
                     input_shape = (128,128,1),
                     kernel_initializer='glorot_uniform'))
    
    model.add(Activation('elu'))
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
    
    model.add(Activation('elu'))
    
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
    
    model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
    
    model.add(Activation('elu'))
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
    
    model.add(Activation('elu'))
    
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
    
    model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
    
    model.add(Activation('elu'))
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
    
    model.add(Activation('elu'))
    
    model.add(BatchNormalization())
    
    model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(2048))
    
    model.add(Activation('elu'))
    
    model.add(BatchNormalization())
    
    model.add(Dropout(0.5))
    
    model.add(Dense(8, activation='softmax'))
         
    
    # compile model
    adam = optimizers.Adam(lr=1e-3, decay=0.95e-3)
    model.compile(optimizer= adam,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    if(multi_gpu == True):
        parallel_model = multi_gpu_model(model, gpus=num_gpus)


    
        parallel_model.compile(optimizer= adam,
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model, parallel_model
    
    
    return model



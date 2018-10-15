#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 09:28:46 2018

@author: zuhayr

Training on dbc multi gpu in docker image

Only saves last model

"""
print('starting')

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, MaxPool2D, Conv2D, Activation, \
    Dense, Flatten, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score
from model_keras_1 import model_from_paper
from DataGenerator import CreateXY_tf, generator, read_val_data
import numpy as np
import os
import sys
import pickle
import time
print(keras.__version__)



os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # specify GPUs to be used
epochs = 400
batch_size = 64
fold_num=3
multi_gpu = True
num_gpus = 2

# Define all paths
if (sys.argv[0] == 'dbc'):
    output_base_dir = '/dbc/output' 
    data_path = '/dbc/data'     
else:
    output_base_dir = './'
    data_path = '/new_mit_data_no_augmentation'
#    data_path = os.getcwd() + '/dummy_data'
    
output_base_dir = '/dbc/output' 
data_path = '/dbc/data'         
print('output_base_dir    ',output_base_dir)
#save_dir = output_base_dir + '/checkpoints/'



def read_data(fold_num):
    
    
    # For ALL data
    image_paths, labels_dict, y, enc = CreateXY_tf(data_path)
    image_paths = np.array(image_paths)
    y = np.array(y)
    
    # Split data using kfoldCrossCorr
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    skf.get_n_splits(image_paths, y)
    
    arr = []
    for train_index, test_index in skf.split(image_paths, y):
        X_train, X_test = image_paths[train_index], image_paths[test_index]
    
        y_train, y_test = y[train_index], y[test_index]
    
        
        print('X_train size = {}, X_test size = {}'.format(len(X_train), len(X_test)))
    
        arr.append([X_train, X_test, y_train, y_test, enc])    
        
    return arr[fold_num]

def create_valid_set(X, y):
    '''
    10% of remaining test set is set as a validation set
    '''
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    sss.get_n_splits(X, y)
        
    for train_index, test_index in sss.split(X, y):
       X_train, X_valid = X[train_index], X[test_index]
       y_train, y_valid = y[train_index], y[test_index]

    return X_train, y_train, X_valid, y_valid

def load_split_dataset(fold_num):
    '''
    param: fold_num - fold number from 0-9 for Kfold cross validations
    Return training, validation and test sets for ECG dataset for a specified fold number
    '''
    print('FOLD NUMBER = ', fold_num)
    

    X_train_temp, X_test, y_train_temp, y_test, enc = read_data(fold_num) # choose fold 0 - 9
    print('X_test size = {}'.format(len(X_test)))
    
    
    # Now that we have train and test set, we have to make valid set
    X_train, y_train, X_valid, y_valid = create_valid_set(X_train_temp, y_train_temp)
    print('X_train size = {}, X_valid size = {}'.format(len(X_train), len(X_valid)))    
    return X_train, X_valid, X_test, y_train, y_valid, y_test 


def run(fold_num):
    start_time = time.time()
    # load model
    model, parallel_model =  model_from_paper(multi_gpu, num_gpus) 
    
    
    # load the dataset
    X_train, X_val, X_test, y_train, y_val, y_test  = load_split_dataset(fold_num)
    
    
    # create data generators
    # note that the full label dict is picklepicklepassed every time, the calculated splits
    # are only used for compatability
    image_paths, labels, y, enc = CreateXY_tf(data_path)
    
    train_generator = generator(X_train, labels, batch_size)
    test_generator = generator(X_test, labels, batch_size)
    
    X_val_imgs = read_val_data(X_val)
    
    
    #callbacks = [# save checkpoints
    #            ModelCheckpoint(filepath= save_dir + 'weights.h5',
    #                            save_best_only=True, monitor='val_acc', mode = 'max', save_weights_only=True)
    #            ]
                
    print('debug1')
    history = parallel_model.fit_generator(train_generator,  # Features, labels
                                  steps_per_epoch=len(X_train)//batch_size,
                                  epochs=epochs,
                                  validation_data=(X_val_imgs, y_val),
                                  verbose=1 
                                  #callbacks=callbacks
                                  )  
                                  
    print('debug2')
    # save training history
    with open(output_base_dir + '/trainHistoryDict_{}.pkl'.format(fold_num), 'wb') as file_1:
        pickle.dump(history.history, file_1)
    #save model
    model.save_weights(output_base_dir + '/my_model_weights.h5')
    print('model saved')
    # Testing on Training Set
    
    # Predict classes
    #score = parallel_model.evaluate(X_test_imgs, y_test)
    predictions = model.predict_generator(generator=test_generator, 
                                          use_multiprocessing=False, 
                                          workers=6, 
                                          steps = len(X_test)//batch_size)
        
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1) 
    
    # Get ground-truth classes and class-labels
    true_classes = y_test[0:len(predictions)]    
    
    report = classification_report(true_classes, predicted_classes)
    print(report) 
    
    end_time = time.time()
    print('Total processing time:', end_time - start_time)
    print(".............END..............")
    
        
if __name__ == "__main__":
    run(fold_num)    

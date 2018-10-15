import os
import glob
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder


def CreateXY(data_path = os.getcwd() + '/data'):    
    """
    
        Create array of file paths X and dict of file paths to labels
        
        params: data_path - directory to ECG dataset, with a separate directory
                for each class
        returns: X - Array of file paths
                 y - Array of labels
                 dict_ - dictionary with keys as X entries and values as 
                 corresponding y entires
        
    """
    cats = {'NOR':0,'APC':1,'VEB':2,'VFW':3,'RBB':4,'PVC':5,'PAB':6,'LBB':7}

    # paths to all images
    X = []
    X = glob.glob(data_path + '/*/*.png')

    y = []
    for file in X:
        parent_dir = os.path.abspath(os.path.join(file, os.pardir))
        y.append(parent_dir.rsplit('/', 1)[1])

    # encode labels array    
    for i in range(len(y)):
        try:
            y[i] = cats[y[i]]
        except KeyError:
            print('error')  
    dict_ = dict(zip(X, y))
    
    return X, dict_, y


        

def generator(image_paths, labels, batch_size):  
    """
    
    Data generator for training data
    
    """      
         
    while True:
        X = []
        y = []   
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(image_paths),1)[0]
             
            ##load image 
            image = cv2.imread(image_paths[index], cv2.IMREAD_GRAYSCALE)
            image = np.reshape(image, (128, 128, 1))
            X.append(image)
            
            y.append(labels[image_paths[index]]) #add the label for the image   
    
        yield np.array(X), np.array(y)
        
        
        
def read_val_data(image_paths):
    """
    
    params: image_paths - array of image paths
    
    returns:  array of grayscale, 128x128 images
    
    Used since validation data is not being read by the generator.
    It is being read all at once
    
    """
    X = []
    for i in range(len(image_paths)):
         
        ##load image 
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        image = np.reshape(image, (128, 128, 1))
        X.append(image)
        
    return np.array(X)
    


# ONE HOt ENCODED
def CreateXY_tf(data_path = os.getcwd() + '/data'):    
    """
    
        Create array of file paths X and dict of file paths to labels
        
        params: data_path - directory to ECG dataset, with a separate directory
                for each class
        returns: X - Array of file paths
                 y - Array of labels - ONE HOT ENCODED
                 dict_ - dictionary with keys as X entries and values as 
                 corresponding y entires
        
    """
    cats = {'NOR':0,'APC':1,'VEB':2,'VFW':3,'RBB':4,'PVC':5,'PAB':6,'LBB':7}

    # paths to all images
    X = []
    X = glob.glob(data_path + '/*/*.png')

    y = []
    for file in X:
        parent_dir = os.path.abspath(os.path.join(file, os.pardir))
        y.append(parent_dir.rsplit('/', 1)[1])
        

    # encode labels array    
    for i in range(len(y)):
        try:
            y[i] = [cats[y[i]]]
        except KeyError:
            print('error')  
            
    # create one hot encoding object to encode y during training
    enc = OneHotEncoder()
    enc.fit(y)

    
    # create dictionary, linking X to y
    dict_ = dict(zip(X, y))
    
    return X,dict_, y, enc


def read_val_data_tf(image_paths):
    """
    
    params: image_paths - array of image paths
    
    returns:  array of grayscale, 128x128 images
    
    """
    X = []
    for i in range(len(image_paths)):
         
        ##load image 
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        image = np.reshape(image, (1, 128, 128, 1))
        X.append(image)
        
    return np.array(X)

def read_train_data_tf(image_paths):
    """
    
    params: image_paths - array of image paths
    
    returns:  array of grayscale, 128x128 images
    
    """
    X = []
    for i in range(len(image_paths)):

        ##load image 
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        image = np.reshape(image, (128, 128, 1))
        X.append(image)
        
    return np.array(X)


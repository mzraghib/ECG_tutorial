import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import wfdb


def get_paths(mit_bih_path = '/home/zuhayr/DBC/ECG/data/MIT_BIH_arrhythmia_database/all'):
    """ 
        Get paths for data record ids in the common directory
        
        params: mit_bih_path - path to all files of mit-bih dataset in the same directory
        returns: paths -  array of paths
                          (single path for 3 file types since no extension
                          needed for wfdb library)
    """
    paths = glob.glob(mit_bih_path + '/*.atr')

    # remove extension, the wfdb library selects the extension itself
    paths = [path[:-4] for path in paths]
    paths.sort()

    return paths



def save_imgs(signals,
              category_string = 'NOR',
              record = 100,
              resize = True,
              save_dir = './'):
    '''
    convert segmented signals into images
    
    The images are saved in separate folders by class
    
    note: used in segmentation() function below
    '''
    fig = plt.figure(frameon=False)

    for count, i in enumerate(signals):
        plt.plot(i) 
        plt.xticks([]), plt.yticks([])
        
        # remove spines / boundaries of plot
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
       
        # save as image
        directory = save_dir +  category_string
        if not os.path.exists(directory):
            os.makedirs(directory)  
            
        filename = directory + '/{}_{}'.format(record, count) + '.png'

        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        
        if(resize == True):
            # no noticable difference between different interpolation methods
            im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
            
        cv2.imwrite(filename, im_gray)
        
        plt.clf() # clear all plots

          


'''
TRAINING
'''
def segmentation(path, 
                 category = ['N'], 
                 category_string = 'NOR',
                 save_dir = './'):
    '''
    Segments the beats of a given category then saves as images using save_imgs()
    
    
    default setting for normal signals ...
    '''
    category_signals = []
        
    signals, fields = wfdb.rdsamp(path, channels = [0]) # get signal data

    ann = wfdb.rdann(path, 'atr') # annotation file

    ids = np.in1d(ann.symbol, category) # ids of the specified category
    imp_beats = ann.sample[ids] # only the beats for the ids above
    beats = (ann.sample) 
    for i in imp_beats:
        beats = list(beats)
        j = beats.index(i)
        if(j!=0 and j!=(len(beats)-1)): # exclude first and last beat
            curr_peak = beats[j] # current Q-wave peak beign considered
            prev_peak = beats[j-1] # previous beat among all beats
            next_peak = beats[j+1] # next beat among all beats
            
            diff1 = abs(prev_peak - curr_peak)//2 # not in line with paper
            diff2 = abs(next_peak - curr_peak)//2   
        
            
            category_signals.append(signals[prev_peak+20: next_peak-20, 0])
            
            
            
    # now with the segmented signals at hand, save as images
    save_imgs(signals = category_signals, 
              category_string = category_string, 
              record =  int(ann.record_name), 
              resize = True,
              save_dir = save_dir)
        

        
'''

TESTING - FOR SEGMENTING UNLABELLED ECG DATA, NOT USED!

'''
#import biosppy
#import wfdb
#def segment(ecg_signals):
#    '''
#    segment ecg signals based on R-peaks
#    '''
#    data = np.array(ecg_signals)
#    signals = []
#    count = 1
#    peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
#    for i in (peaks[1:-1]):
#        diff1 = abs(peaks[count - 1] - i)
#        diff2 = abs(peaks[count + 1]- i)
#        x = peaks[count - 1] + diff1//2
#        y = peaks[count + 1] - diff2//2
#        signal = data[x:y]
#        signals.append(signal)
#        count += 1
#    return signals



def test_segmentation():
    
    
    path = paths[0]
    category = ['E'], 
    category_string = 'VEB',
    save_dir = '/home/zuhayr/Desktop/new_mit_data'
                 
                 
                 
                 
                 
    plt.figure(figsize=(15, 7))
    plt.plot(signals[:1000]) 

                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
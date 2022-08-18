# Tools for processing time series data and infernecing 
# Wathela Alhassa
#07-May-2022

from tensorflow.keras.models import load_model 
from tensorflow.keras import layers, losses
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_recall_fscore_support
import tensorflow as tf
from scipy.signal import stft,istft
from astropy.coordinates import Distance
from astropy.cosmology import Planck15
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from itertools import islice
import librosa
from tqdm import tqdm
import itertools
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp
import random
import shutil
import glob
import os

def load_classifier(mdl_path):
    model = load_model(mdl_path)
    optimizer = RMSprop(learning_rate=0.0001,rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=1.)#SGD(lr=0.0001, decay=1e-6, momentum=0.9,clipnorm=1, nesterov=True)
    model.compile(loss='categorical_crossentropy',#'binary_crossentropy',categorical
                  optimizer=optimizer,
                  metrics=['accuracy'])#'binary_accuracy'])
    return model

def read_in_chunks(file_object, chunk_size=5*16384*12*23):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.readlines(chunk_size)
        if not data:
            break
        yield data


def split_list(ts, n):

    """Yield successive n-sized chunks from the time series.
    If the length of the last chunk < n, returns last n values."""
    rows = n 
    if len(ts) <= rows:
        
        yield ts   
    else:
        for i in range(0, len(ts), rows):
#             if i - rows <= 0:
            sub = ts[i:i + rows]

#             elif i - rows > 0:
#                 sub = ts[i-16384:i + n - 16384]
                
#             if len(sub) < rows:
#                 sub = ts[-rows:]    
            yield sub
            
def ts_to_img(ts,et_fs=16384):
    nfft = 2048*2
    f, t, Sxx = sp.signal.spectrogram(x=np.array(ts),fs=et_fs, nfft=nfft,mode='magnitude')
    
    img = scale_minmax(Sxx)
    imgf = np.flip(img, axis=0)
    img = imgf[-42:,:]# 300 hz
    img = img.reshape(img.shape[0], img.shape[1],1)
    return img

def scale_minmax(X, mn=0.0, mx=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (mx - mn) + mn
    return X_scaled

def scale(x, out_range=(0, 255)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def detect(img,detector):
    mag_scaled = img#scale(img)
#     print(mag_scaled.min(),mag_scaled.max())
#     mag_rgb_reshaped = np.reshape(mag_scaled,(mag_scaled.shape[0],mag_scaled.shape[1],3))
    mag_rgb_reshaped = np.expand_dims(mag_rgb_reshaped, axis=0)#mag_rgb_reshaped

    prd = np.argmax(detector.predict(mag_rgb_reshaped), axis=1)
    
    return prd[0]

def gray2rgb(gray):
    gray = scale(gray)
    if gray.shape != (42,365,1):
        gray = np.resize(gray,[42,365,1])
    #d = np.zeros((42,365)) #np.stack((gray,d,d),axis=2)
    return np.concatenate((gray,)*3, axis=-1)

def load_image(filename):
    raw = tf.io.read_file(filename)
    img = tf.image.decode_png(raw, channels=3)
    reshaped_img = np.expand_dims(img, axis=0)
    
    return reshaped_img

def find_class(result):
    if result >= 0.5:
        return 1
    else:
        return 0

def get_lines_iterator(filename, n=5*16384):
    with open(filename) as fp:
        while True:
            lines = list(islice(fp, n))
            if lines:
                yield lines
            else:
                break


def save_report(outfile,window_id,cls,window_t0,window_tend,idx0,idxend):
  
    with open(outfile, 'a') as f:
        with redirect_stdout(f):
            print("{},{},{},{},{},{}".format(window_id,cls,window_t0,window_tend,idx0,idxend))
    f.close()



def ET_BBH_detector(ts_file,detector,outfile):
    window_t = 0
    window_id = 0
    idx = 0
    inj_list = [] # A list to count # of injected spectorgrams
    noise_list = [] # A list to count # of only noise spectrograms
    
    for ts in get_lines_iterator(ts_file):
        ts_list = [float(val) for val in ts]

        img = ts_to_img(ts_list)
        img_rgb = gray2rgb(img)
        reshaped_img_rgb = np.expand_dims(img_rgb, axis=0)
        prd = detector.predict(reshaped_img_rgb)

        #             plt.imshow(np.flipud(img),origin = 'lower')
        #             plt.xticks(np.arange(0,365,73), labels=list(range(5)))
        #             plt.yticks(np.arange(0,42,14),list(range(0,300,100)))
        #             plt.xlabel('Time[seconds]')
        #             plt.ylabel('Freq [Hz]')
        #             plt.show()
        #             print(img.shape)
        #             print(70*"-")

        cls = find_class(prd[0])
        
        
        print(f"Scaning...")
        print(f"Window duration: from {window_t}s to {window_t+5}s")
        cls_label = None
        if cls==0:
            print("A BBH merger detected!")
            inj_list.append(1)
            cls_label="inj"

        elif cls ==1:
            print("No merger was detected!")
            noise_list.append(1)
            cls_label="noise"
            
          
        save_report(outfile, window_id,cls_label, window_t,window_t+5, idx, idx+5*16384)   
            
        window_t +=5
        window_id +=1
        idx += 5*16384#*12*23
        
    print()
    print("Report:")
    print(f"{len(inj_list)} sources were detected.")
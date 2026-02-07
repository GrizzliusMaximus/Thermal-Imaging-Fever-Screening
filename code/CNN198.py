from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import codecs, json 

import cv2
import matplotlib.pyplot as plt

from sklearn import mixture
from scipy import linalg as la
from scipy import stats as st
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf

def read_tempdata(rn, path):
    arr = np.zeros([rn,8,8])
    with open(path) as f:
        for k in range(rn):
            for i in range(8):
                line = f.readline()
                t = [float(x) for x in line.split()]
                arr[k,i] = t
            line = f.readline()
    return arr

def read_gtempdata(rn,gpath,num):
    garr = np.zeros([num,rn,8,8])
    for n in range(num):
        path = gpath + str(n+1) +'.txt'
        garr[n] = read_tempdata(rn, path)
    return garr

def thermal_interpolate(garr,oy,ox,ny,nx):
    interpol = np.zeros((ny,nx,8,8))
    garr2 = garr.reshape((oy,ox,8,8))
    for i in range(ny):
        for j in range(nx):
            y2 = max(0,math.ceil(i*(oy-1)/(ny-1)))
            y1 = y2-1
            x2 = max(0,math.ceil(j*(ox-1)/(nx-1)))
            x1 = x2-1
            ly = i*(oy-1)/(ny-1) - y1
            lx = j*(ox-1)/(nx-1) - x1
            inter1 = garr2[y1,x1]*(1-ly)+garr2[y2,x1]*ly
            inter2 = garr2[y1,x2]*(1-ly)+garr2[y2,x2]*ly
            interpol[i,j] = inter1*(1-lx)+inter2*lx
    return interpol

def output_interpolate(temp_path, ln2):
    interpol = np.zeros((4,ln2**2,100,8,8))
    ln = 6
    bg_max = 0
    for i in range(4):
        datapath = 'train/' + temp_path + '/output-' + str(i+1)+ '/'

        garr = read_gtempdata(100,datapath,ln**2)

        interpol[i] = thermal_interpolate(garr,ln**2,100,ln2**2,100)
        ln -= 1
    return interpol


def load_temperature():
    ln = 6
    x_train = np.zeros((4,4,ln**2,100,8,8))
    temp_path = np.array(['hot','warm','semicold','cold'])
    for i in range(4):
        x_train[i] = output_interpolate(temp_path[i],ln)
    x_train = x_train.reshape(-1,8,8,1) 
    
    file_path = "amax.json"
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    arr_list= json.loads(obj_text)
    
    y_train = np.ones((4,4,4,ln**2,100))
    y_train[0] = (np.ones((100,4,4,ln**2))*np.array(arr_list).reshape(4,4,ln**2)).transpose(1,2,3,0)
    pos_x = np.array([np.arange(1/12,1,1/6),]*6)
    pos_y = pos_x.T
    y_train[1] = (np.ones((4,ln**2,100,4))*np.linspace(1,4,4)).transpose(0,3,1,2)
    y_train[2] = (np.ones((4,4,100,ln**2))*pos_x.reshape(-1)).transpose(0,1,3,2)
    y_train[3] = (np.ones((4,4,100,ln**2))*pos_y.reshape(-1)).transpose(0,1,3,2)
    y_train = y_train.reshape(4,-1).T
    
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    
    return x_train[indices[:-100]], y_train[indices[:-100]], x_train[indices[-100:]], y_train[indices[-100:]]

x_train, y_train, x_test, y_test = load_temperature()

input_shape = (8,8,1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# use functional API to build cnn layers
inputs = Input(shape=input_shape)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
# image to vector before connecting to dense layer
y = Flatten()(y)
# dropout regularization
y = Dropout(dropout)(y)
outputs = Dense(4, activation='sigmoid')(y)

# build the model by supplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)
# network model in text
model.summary()


# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer='adam',
              metrics=['accuracy'])

# train the model with input images and labels
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=20,
          batch_size=batch_size)
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization

import cv2
import os
import numpy as np

def build_model(batch_size=512, im_shape=(128, 128, 1)):
    
#batch_size=512
#im_shape=(*X[0].shape, 1)

# im_shape = (128, 128, 1)
    cnn_model= Sequential([
        Conv2D(filters=32, kernel_size=7, activation='relu', input_shape= im_shape),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=48, kernel_size=7, activation='relu', input_shape= im_shape),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=54, kernel_size=7, activation='relu', input_shape= im_shape),
        MaxPooling2D(pool_size=2),
        
        Flatten(),
        Dense(2024, activation='relu'),
         Dropout(0.4),
        Dense(1024, activation='relu'),
          Dropout(0.4),
        Dense(512, activation='relu'),
          Dropout(0.4),
        #29 is the number of outputs
        Dense(29, activation='softmax')  
    ])

    cnn_model.compile(
        loss='sparse_categorical_crossentropy',#'categorical_crossentropy',
        optimizer=Adam(lr=0.0001),
        metrics=['accuracy']
    )
    return cnn_model

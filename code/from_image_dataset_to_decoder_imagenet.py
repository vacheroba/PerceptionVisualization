# Takes the dataset_encoder_imagenet_onlyimg and computes dataset_encoder_imagenet

import pandas as pd
import numpy as np
from PIL import Image
import os
import importdataset
from keras import applications, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from keras import metrics
from keras.models import Sequential
import keras.backend as K
import keras
from bpmll import bp_mll_loss
import utils
import h5py
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)

basepath = os.getcwd()
datasetpath = os.path.join(basepath, "../datasets/dataset_encoder_imagenet_onlyimg.h5")

base_model = tf.keras.applications.EfficientNetB7(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)

encoder = keras.Model(base_model.input, base_model.get_layer("top_activation").output)
del base_model

encoder.summary()

num_images = 4000

hf = h5py.File(datasetpath, 'r')

X_train = hf.get('X_train').value
E_train = np.zeros([num_images, 19, 19, 2560], dtype=np.float32)
print("Loading images")

for i in range(0, num_images):
    if i % 100 == 0:
        print(i)
    E_train[i, :, :, :] = encoder.predict(X_train[i:i+1, :, :, :])

# Gets outputs of modified model
print("Predicting")
# E_train = encoder.predict(X_train)

print(E_train.shape)
print(X_train.shape)

# Saves in h5
print("Saving result")
datasetpath = os.path.join(basepath, "../datasets/dataset_encoder_imagenet.h5")
hf = h5py.File(datasetpath, 'w')
hf.create_dataset('X_train', data=X_train)
hf.create_dataset('E_train', data=E_train)
hf.close()


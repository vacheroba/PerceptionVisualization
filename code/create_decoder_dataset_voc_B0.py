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

# Loads model from classifier checkpoint
print("Loading model")
basepath = os.getcwd()
modelpath = os.path.join(basepath, "../models/classifier")

classifier = tf.keras.applications.EfficientNetB0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)

encoder = keras.Model(classifier.input, classifier.get_layer("top_activation").output)
del classifier

# Loads dataset
print("Loading dataset")
X_train, X_test, Y_train, Y_test = importdataset.load_dataset()

X_train = X_train*255
X_test = X_test*255

# Gets outputs of modified model
print("Predicting")
E_train = encoder.predict(X_train)
del X_train
E_test = encoder.predict(X_test)
del X_test

# Saves in h5
print("Saving result")
datasetpath = os.path.join(basepath, "../datasets/dataset_encoder_voc_B0.h5")
hf = h5py.File(datasetpath, 'w')
hf.create_dataset('E_train', data=E_train)
hf.create_dataset('E_test', data=E_test)
hf.close()

















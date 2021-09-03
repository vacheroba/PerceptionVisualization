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

# Loads model from classifier checkpoint
print("Loading model")
basepath = os.getcwd()
modelpath = os.path.join(basepath, "../models/classifier")
model = keras.models.load_model(modelpath, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss})

# Creates encoder by removing last layers from the classifier
print("Transforming model")
feature_layer = "global_average_pooling2d"
encoder = keras.Model(inputs=model.input, outputs=model.get_layer(feature_layer).input)
del model

# Loads dataset
print("Loading dataset")
X_train, X_test, Y_train, Y_test = importdataset.load_dataset()

# Gets outputs of modified model
print("Predicting")
E_train = encoder.predict(X_train)
E_test = encoder.predict(X_test)

# Saves in h5
print("Saving result")
datasetpath = os.path.join(basepath, "../datasets/dataset_encoder.h5")
hf = h5py.File(datasetpath, 'w')
hf.create_dataset('E_train', data=E_train)
hf.create_dataset('E_test', data=E_test)
hf.close()

















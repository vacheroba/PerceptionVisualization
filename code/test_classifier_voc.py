import pandas as pd
import numpy as np
from PIL import Image
import os
import importdataset
from keras import applications, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Flatten, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from keras import metrics
from keras import losses
from keras.models import Sequential
import keras.backend as K
from PIL import Image
from PIL import ImageTk, ImageWin
import tkinter
import keras
from bpmll import bp_mll_loss
import utils
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

basepath = os.getcwd()
classifier_path = os.path.join(basepath, "../models/classifier")
main_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")

classifier = keras.models.load_model(classifier_path, custom_objects={"bp_mll_loss": bp_mll_loss,
                                                            "euclidean_distance_loss": utils.euclidean_distance_loss})

# Load targets (The targets for the decoder are the original inputs, X in main dataset)
hf = h5py.File(main_dataset_path, 'r')
X_test = hf['X_Test']
Y_test = hf['Y_Test']

for i in range(0, 20):
    print("\n\n")
    sample = X_test[i:i+1, :, :, :]
    target = Y_test[i, :]

    print("Targets")
    accepted = []
    count = 0
    for elem in importdataset.CLASS_NAMES:
        if target[count] > 0.1:
            accepted.append((elem, target[count]))
        count += 1
    print(accepted)

    print("Model prediction")
    classes = classifier.predict(sample).squeeze()
    accepted = []
    count = 0
    for elem in importdataset.CLASS_NAMES:
        if classes[count] > 0.1:
            accepted.append((elem, classes[count]))
        count += 1

    print(accepted)
    input("Any key to continue")

































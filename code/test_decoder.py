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
decoder_path = os.path.join(basepath, "../models/decoder")
classifier_path = os.path.join(basepath, "../models/classifier")
main_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")
encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder.h5")

decoder = keras.models.load_model(decoder_path, custom_objects={"bp_mll_loss": bp_mll_loss,
                                                            "euclidean_distance_loss": utils.euclidean_distance_loss})

classifier = keras.models.load_model(classifier_path, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss})

decoder.summary()
classifier.summary()

# Load targets (The targets for the decoder are the original inputs, X in main dataset)
hf_main = h5py.File(main_dataset_path, 'r')
# Y_train = hf.get('X_Train').value
X_test = hf_main['X_Test']
Y_test = hf_main['Y_Test']

hf_enc = h5py.File(encoder_dataset_path, 'r')
# X_train = hf.get('E_train').value
E_test = hf_enc['E_test']

root = tkinter.Tk()
root.geometry('1000x1000')
canvas = tkinter.Canvas(root, width=999, height=999)
canvas.pack()

for i in range(40, 250):
    reconstructed = ((decoder.predict(E_test[i:i+1, :, :, :]))*255).squeeze().astype(np.uint8)
    original = (X_test[i:i+1, :, :, :]*255).squeeze().astype(np.uint8)
    res = np.concatenate((original, reconstructed), axis=1)

    print("\n\n")
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
    classes = classifier.predict(X_test[i:i+1, :, :, :]).squeeze()
    # print(classes)
    accepted = []
    count = 0
    for elem in importdataset.CLASS_NAMES:
        if classes[count] > 0.1:
            accepted.append((elem, classes[count]))
        count += 1
    image = Image.fromarray(res)
    image = ImageTk.PhotoImage(image)
    imagesprite = canvas.create_image(400, 400, image=image)
    root.update()
    print(accepted)
    input("Any key to continue")




































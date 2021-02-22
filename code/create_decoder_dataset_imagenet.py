# Takes the EfficientNetB7 last convolutional results and creates the dataset in dataset_encoder_imagenet
# to then be used to train the decoder

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
tf.config.experimental.set_memory_growth(physical_devices[0], True)

basepath = os.getcwd()
imagenetpath = os.path.join(basepath, "../datasets/ImageNet/imagenet_object_localization_patched2019.tar/ILSVRC/Data/CLS-LOC/test")

base_model = tf.keras.applications.EfficientNetB0(
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

num_images = 20000

# For B7
# X_train = np.zeros([num_images, 600, 600, 3], dtype=np.uint8)
# E_train = np.zeros([num_images, 19, 19, 2560], dtype=np.float32)

# For B0
X_train = np.zeros([num_images, 224, 224, 3], dtype=np.uint8)
E_train = np.zeros([num_images, 7, 7, 1280], dtype=np.float32)

print("Loading images")
image_paths = os.listdir(imagenetpath)

# img_size = 600
img_size = 224

for i in range(0, num_images):
    if i % 100 == 0 and i > 0:
        E_train[i-100:i, :, :, :] = encoder.predict(X_train[i-100:i, :, :, :])
        print(i)
    img = Image.open(os.path.join(imagenetpath, image_paths[i]))
    img = img.resize((img_size, img_size))
    imgarray = np.array(img)
    if imgarray.shape == (img_size, img_size):
        h = np.zeros([img_size, img_size, 3]).astype(np.uint8)
        h[:, :, 0] = imgarray
        h[:, :, 1] = imgarray
        h[:, :, 2] = imgarray
        imgarray = h
    X_train[i, :, :, :] = imgarray
    # E_train[i, :, :, :] = encoder.predict(X_train[i:i+1, :, :, :])

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
























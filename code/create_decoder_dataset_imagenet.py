# Creates a dataset of encodings of ImageNet images plus their original form

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
import gc
import random

random.seed(2222)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

basepath = os.getcwd()
imagenetpath = os.path.join(basepath, "../datasets/ImageNet/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test")
modelpath = os.path.join(basepath, "../models/classifier")

model = keras.models.load_model(modelpath, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss})

# Creates encoder by removing last layers from the classifier
print("Transforming model")
feature_layer = "global_average_pooling2d"
encoder = keras.Model(inputs=model.input, outputs=model.get_layer(feature_layer).input)
del model

num_images = 8000

X_train = np.zeros([num_images, 224, 224, 3], dtype=np.float32)
E_train = np.zeros([num_images, 7, 7, 2048], dtype=np.float32)

print("Loading images, total")
image_paths = os.listdir(imagenetpath)
random.shuffle(image_paths)
print(len(image_paths))

# img_size = 600
img_size = 224

for i in range(0, num_images):
    img = Image.open(os.path.join(imagenetpath, image_paths[i])).convert('RGB')
    img = img.resize((img_size, img_size))
    imgarray = np.array(img).astype(np.float32)/255.0
    # if imgarray.shape == (img_size, img_size):
    #     h = np.zeros([img_size, img_size, 3]).astype(np.uint8)
    #     h[:, :, 0] = imgarray
    #     h[:, :, 1] = imgarray
    #     h[:, :, 2] = imgarray
    #     imgarray = h
    X_train[i, :, :, :] = imgarray

    if i % 100 == 99:
        E_train[i-99:i+1, :, :, :] = encoder.predict(X_train[i-99:i+1, :, :, :])
        print(i)
    # E_train[i, :, :, :] = encoder.predict(X_train[i:i+1, :, :, :])

# Rescales images

# Gets outputs of modified model
print("Predicting")
# E_train = encoder.predict(X_train)

print(E_train.shape)
print(X_train.shape)

# Saves in h5
print("Saving result")
datasetpath = os.path.join(basepath, "../datasets/dataset_encoder_imagenet.h5")
hf = h5py.File(datasetpath, 'w')
hf.create_dataset('E_train', data=E_train)
hf.close()

datasetpath = os.path.join(basepath, "../datasets/dataset_imagenet.h5")
hf = h5py.File(datasetpath, 'w')
hf.create_dataset('X_Train', data=X_train)
hf.close()
























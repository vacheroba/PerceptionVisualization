# Takes the standard resnet trained for pascal VOC and retrains the last layer
# Reconstructions are untouched but the network can in this way be used for other tasks
# In this case, the task is that of traffic sign classification from TSRD

import pandas as pd
import numpy as np
from PIL import Image
import os
import importdataset
from keras import applications, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, AveragePooling2D, Flatten, BatchNormalization, SpatialDropout2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from keras import metrics
from keras.models import Sequential
import keras.backend as K
from bpmll import bp_mll_loss
import utils
import tensorflow as tf
import h5py
import math
from keras.losses import binary_crossentropy
import keras

basepath = os.getcwd()
tsrd_path = "N:/PycharmProjects/scratchthat/datasets/tsrd/classes"
classifier_path = os.path.join(basepath, "../models/classifier")

BATCH_SIZE = 64  # 16 for my pc
NUM_EPOCHS = 100
VALID_SPLIT = 0.15

physical_devices = tf.config.list_physical_devices('GPU')

for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)

img_height, img_width = 224, 224
num_classes = 58


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    tsrd_path,
    validation_split=VALID_SPLIT,
    subset="training",
    seed=2222,
    image_size=(img_width, img_height),
    batch_size=BATCH_SIZE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

base_model = keras.models.load_model(classifier_path, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss, "rgb_ssim_loss": utils.rgb_ssim_loss})
encoder = keras.Model(base_model.input, base_model.get_layer("global_average_pooling2d").input)
encoder.trainable = False

model = Sequential()
model.add(encoder)

model.add(SpatialDropout2D(rate=0.5))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


model.fit(dataset, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

basepath = os.getcwd()
modelpath = os.path.join(basepath, "../models/classifier_tsrd")
model.save(modelpath)

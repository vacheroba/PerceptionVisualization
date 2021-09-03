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
import keras.layers as layers
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

basepath = os.getcwd()
classifier_path = os.path.join(basepath, "../models/classifier_flowers")

base_model = keras.models.load_model(classifier_path)
encoder = keras.Model(base_model.input, base_model.get_layer("embedding").output)

# These are the dataset in the form to train the encoder
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Now lambda the datasets to have the required embedding dataset to train the decoder
train_emb = train_ds.map(lambda x, y: (encoder(x), x))
val_emb = val_ds.map(lambda x, y: (encoder(x), x))

# Make decoder
decoder = tf.keras.Sequential()
decoder.add(layers.Conv2DTranspose(input_shape=(9, 9, 32), filters=64, kernel_size=3, padding="same", strides=5))
decoder.add(layers.BatchNormalization())
decoder.add(layers.Conv2D(filters=32, padding="same", kernel_size=3, activation="sigmoid"))
decoder.add(layers.Conv2DTranspose(filters=32, kernel_size=2, padding="same", strides=2))
decoder.add(layers.BatchNormalization())
decoder.add(layers.Conv2D(filters=32, padding="same", kernel_size=3, activation="sigmoid"))
decoder.add(layers.Conv2DTranspose(filters=32, kernel_size=2, padding="same", strides=2))
decoder.add(layers.BatchNormalization())
decoder.add(layers.Conv2D(filters=32, padding="same", kernel_size=3, activation="sigmoid"))
decoder.add(layers.Conv2D(filters=3, padding="same", kernel_size=3, activation="sigmoid"))

AUTOTUNE = tf.data.AUTOTUNE

train_emb = train_emb.cache().prefetch(buffer_size=AUTOTUNE)
val_emb = val_emb.cache().prefetch(buffer_size=AUTOTUNE)


decoder.compile(
  optimizer='adam',
  loss='mse',
  metrics=[tf.keras.metrics.MeanSquaredError()])

decoder.fit(
  train_emb,
  validation_data=val_emb,
  epochs=100
)

basepath = os.getcwd()
modelpath = os.path.join(basepath, "../models/decoder_flowers")
decoder.save(modelpath)

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
decoder_path = os.path.join(basepath, "../models/decoder_flowers")

base_model = keras.models.load_model(classifier_path)
encoder = keras.Model(base_model.input, base_model.get_layer("embedding").output)

# These are the dataset in the form to train the encoder
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Now lambda the datasets to have the required embedding dataset to train the decoder
train_emb = train_ds.map(lambda x, y: (encoder(x), x))
val_emb = val_ds.map(lambda x, y: (encoder(x), x))

dataset = train_ds

decoder = keras.models.load_model(decoder_path)

for images, labels in dataset:
    for i in range(batch_size):
        reconstruction = (np.array(decoder(encoder(images[i:i+1])))*255).reshape([180, 180, 3]).astype(np.uint8)
        Image.fromarray(reconstruction).show()
        input("Wait")

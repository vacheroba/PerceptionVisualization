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
import keras
from bpmll import bp_mll_loss
import utils
import h5py
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow_io as tfio
import math
import gc

basepath = os.getcwd()
encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder_voc_B0.h5") # "../datasets/dataset_encoder_imagenet_rescaled.h5"
voc_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")

BATCH_SIZE = 16
NUM_EPOCHS = 10

with h5py.File(encoder_dataset_path, 'r') as hf, h5py.File(voc_dataset_path, 'r') as voc:
    NUM_IMAGES = hf["E_train"].shape[0]

physical_devices = tf.config.list_physical_devices('GPU')

for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)


def generator():
    with h5py.File(encoder_dataset_path, 'r') as hf, h5py.File(voc_dataset_path, 'r') as voc:
        for i in range(0, NUM_IMAGES-BATCH_SIZE, BATCH_SIZE):
            # yield tf.convert_to_tensor(hf["E_train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32), tf.convert_to_tensor(hf["X_train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32)
            yield tf.convert_to_tensor(hf["E_train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32), tf.convert_to_tensor(voc["X_Train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32)


ds_counter = tf.data.Dataset.from_generator(generator, output_signature=(
         tf.TensorSpec(shape=(BATCH_SIZE, 7, 7, 1280), dtype=tf.float32),
         tf.TensorSpec(shape=(BATCH_SIZE, 224, 224, 3), dtype=tf.float32)))

ds_counter = ds_counter.repeat(NUM_EPOCHS)

# Build a reversed VGG16 for decoding
model = Sequential()
# First block (pool->conv->conv->conv)
model.add(Conv2DTranspose(input_shape=(7, 7, 1280), filters=512, kernel_size=2, padding="same", strides=2))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
# Second block (pool->conv->conv->conv)
model.add(Conv2DTranspose(filters=512, kernel_size=2, padding="same", strides=2))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
# Third block (pool->conv->conv->conv)
model.add(Conv2DTranspose(filters=256, kernel_size=2, padding="same", strides=2))
# For B7 you need to add a valid here instead of a same to reduce the size and finally get 600*600
# model.add(Conv2D(filters=256, padding="valid", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
# Fourth block (pool->conv->conv)
model.add(Conv2DTranspose(filters=128, kernel_size=2, padding="same", strides=2))
model.add(Conv2D(filters=128, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=128, padding="same", kernel_size=3, activation="relu"))
# Fifth block (pool->conv->conv)
model.add(Conv2DTranspose(filters=64, kernel_size=2, padding="same", strides=2))
model.add(Conv2D(filters=64, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=64, padding="same", kernel_size=3, activation="relu"))
# Output (conv)
model.add(Conv2D(filters=3, padding="same", kernel_size=3, activation="sigmoid"))
# model.add(Rescaling(255.0, offset=0.0))

model.summary()


# Load targets (The targets for the decoder are the original inputs)
# hf = h5py.File(encoder_dataset_path, 'r')
# Y_train = (hf.get('X_train').value.astype(np.float32))/255.0
# X_train = hf.get('E_train').value
# hf.close()

hf = h5py.File(encoder_dataset_path, 'r')

# Y_train = tfio.IODataset.from_hdf5(encoder_dataset_path, dataset='/X_train')
# X_train = tfio.IODataset.from_hdf5(encoder_dataset_path, dataset='/E_train')
# dataset = tf.data.Dataset.zip((X_train, Y_train)).batch(64, drop_remainder=True)

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=[losses.binary_crossentropy, utils.euclidean_distance_loss])

# model.fit(X_train, Y_train, epochs=100, batch_size=64)
# for i in range(0, math.floor(NUM_EPOCHS/10)):
#    model.fit(ds_counter, epochs=10, batch_size=BATCH_SIZE, steps_per_epoch=math.floor(NUM_IMAGES/BATCH_SIZE))
#    gc.collect()
model.fit(ds_counter, epochs=100, batch_size=BATCH_SIZE, steps_per_epoch=math.floor(NUM_IMAGES/BATCH_SIZE))

modelpath = os.path.join(basepath, "../models/decoder_imagenet_rescaled")
model.save(modelpath)

# preds = model.evaluate(X_train, Y_train)
# print("Loss = " + str(preds[0]))
# print("Test Accuracy = " + str(preds[1]))

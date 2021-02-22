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

physical_devices = tf.config.list_physical_devices('GPU')

for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)

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
model.add(Rescaling(255.0, offset=0.0))

model.summary()

basepath = os.getcwd()
encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder_imagenet.h5")


# Load targets (The targets for the decoder are the original inputs)
hf = h5py.File(encoder_dataset_path, 'r')
Y_train = hf.get('X_train').value
X_train = hf.get('E_train').value
hf.close()

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=[losses.binary_crossentropy, utils.euclidean_distance_loss])

model.fit(X_train, Y_train, epochs=100, batch_size=64)

modelpath = os.path.join(basepath, "../models/decoder_imagenet")
model.save(modelpath)

preds = model.evaluate(X_train, Y_train)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

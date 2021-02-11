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

# Build a reversed VGG16 for decoding
model = Sequential()
# First block (pool->conv->conv->conv)
model.add(Conv2DTranspose(input_shape=(7, 7, 2048), filters=512, kernel_size=2, padding="same", strides=2))
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

model.summary()

basepath = os.getcwd()
main_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")
encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder.h5")


# Load targets (The targets for the decoder are the original inputs, X in main dataset)
hf = h5py.File(main_dataset_path, 'r')
Y_train = hf.get('X_Train').value
Y_test = hf.get('X_Test').value

# Load inputs(The outputs of the encoder, E in encoder dataset)
hf.close()
hf = h5py.File(encoder_dataset_path, 'r')
X_train = hf.get('E_train').value
X_tTest = hf.get('E_test').value
hf.close()

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=[losses.binary_crossentropy, utils.euclidean_distance_loss])

model.fit(X_train, Y_train, epochs=100, batch_size=64)

modelpath = os.path.join(basepath, "../models/decoder")
model.save(modelpath)

















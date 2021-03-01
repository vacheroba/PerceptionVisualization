import pandas as pd
import numpy as np
from PIL import Image
import os
import importdataset
from keras import applications, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Flatten, Conv2DTranspose, BatchNormalization
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
import math


physical_devices = tf.config.list_physical_devices('GPU')

for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)

# Build a reversed VGG16 for decoding
model = Sequential()
# First block (pool->conv->conv->conv)
model.add(Conv2DTranspose(input_shape=(7, 7, 2048), filters=512, kernel_size=2, padding="same", strides=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
# Second block (pool->conv->conv->conv)
model.add(Conv2DTranspose(filters=512, kernel_size=2, padding="same", strides=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
# Third block (pool->conv->conv->conv)
model.add(Conv2DTranspose(filters=256, kernel_size=2, padding="same", strides=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
# Fourth block (pool->conv->conv)
model.add(Conv2DTranspose(filters=128, kernel_size=2, padding="same", strides=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=128, padding="same", kernel_size=3, activation="relu"))
# Fifth block (pool->conv->conv)
model.add(Conv2DTranspose(filters=64, kernel_size=2, padding="same", strides=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, padding="same", kernel_size=3, activation="relu"))
model.add(Conv2D(filters=64, padding="same", kernel_size=3, activation="relu"))
# Output (conv)
model.add(Conv2D(filters=3, padding="same", kernel_size=3, activation="sigmoid"))

model.summary()

basepath = os.getcwd()
main_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")
encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder.h5")

BATCH_SIZE = 64  # 16 for my pc
NUM_EPOCHS = 500
VALID_SPLIT = 0.15

with h5py.File(encoder_dataset_path, 'r') as enc:
    NUM_IMAGES = enc["E_train"].shape[0]
    print("Dataset info")
    print(enc["E_train"].shape)


def generator():
    with h5py.File(main_dataset_path, 'r') as ds, h5py.File(encoder_dataset_path, 'r') as enc:
        for i in range(0, NUM_IMAGES-BATCH_SIZE, BATCH_SIZE):
            yield tf.convert_to_tensor(enc["E_train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32), tf.convert_to_tensor(ds["X_Train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32)


# For tensorflow 2.4
# ds_counter = tf.data.Dataset.from_generator(generator, output_signature=(
#         tf.TensorSpec(shape=(BATCH_SIZE, 7, 7, 1280), dtype=tf.float32),
#         tf.TensorSpec(shape=(BATCH_SIZE, 224, 224, 3), dtype=tf.float32)))

# For tensorflow 2.3
ds_counter = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32), (tf.TensorShape([BATCH_SIZE, 7, 7, 2048]), tf.TensorShape([BATCH_SIZE, 224, 224, 3])))

ds_counter = ds_counter.shuffle(10, reshuffle_each_iteration=True)
ds_counter = ds_counter.repeat(NUM_EPOCHS)

# Load targets (The targets for the decoder are the original inputs, X in main dataset)
hf = h5py.File(main_dataset_path, 'r')
Y_test = hf.get('X_Test').value

# Load inputs(The outputs of the encoder, E in encoder dataset)
hf.close()
hf = h5py.File(encoder_dataset_path, 'r')
E_test = hf.get('E_test').value
hf.close()

model.compile(optimizer='adam',
              loss=utils.euclidean_distance_loss,  # "binary_crossentropy",
              metrics=[losses.binary_crossentropy, utils.euclidean_distance_loss])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_euclidean_distance_loss', patience=20, restore_best_weights=True)
model.fit(ds_counter, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=math.floor(NUM_IMAGES/BATCH_SIZE), callbacks=[callback], validation_data=(E_test, Y_test))


modelpath = os.path.join(basepath, "../models/decoder")
model.save(modelpath)

preds = model.evaluate(E_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
















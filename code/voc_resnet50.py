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
from bpmll import bp_mll_loss
import utils
import tensorflow as tf
import h5py
import math

basepath = os.getcwd()
voc_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")

BATCH_SIZE = 64  # 16 for my pc
NUM_EPOCHS = 500
VALID_SPLIT = 0.15

with h5py.File(voc_dataset_path, 'r') as voc:
    NUM_IMAGES = voc["X_Train"].shape[0]
    print("Dataset info")
    print(voc["X_Train"].shape)


physical_devices = tf.config.list_physical_devices('GPU')

for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)


def generator():
    with h5py.File(voc_dataset_path, 'r') as voc:
        for i in range(0, NUM_IMAGES-BATCH_SIZE, BATCH_SIZE):
            yield tf.convert_to_tensor(voc["X_Train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32), tf.convert_to_tensor(voc["Y_Train"][i:i + BATCH_SIZE, :], dtype=tf.float32)


# For tensorflow 2.4
# ds_counter = tf.data.Dataset.from_generator(generator, output_signature=(
#         tf.TensorSpec(shape=(BATCH_SIZE, 7, 7, 1280), dtype=tf.float32),
#         tf.TensorSpec(shape=(BATCH_SIZE, 224, 224, 3), dtype=tf.float32)))

# For tensorflow 2.3
ds_counter = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32), (tf.TensorShape([BATCH_SIZE, 224, 224, 3]), tf.TensorShape([BATCH_SIZE, 20])))

ds_counter = ds_counter.shuffle(10, reshuffle_each_iteration=True)
ds_counter = ds_counter.repeat(NUM_EPOCHS)

class_names = importdataset.CLASS_NAMES

with h5py.File(voc_dataset_path, 'r') as voc:
    X_test = tf.convert_to_tensor(voc["X_Test"][:, :, :, :], dtype=tf.float32)
    Y_test = tf.convert_to_tensor(voc["Y_Test"][:, :], dtype=tf.float32)

img_height, img_width = 224, 224
num_classes = 20

base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False,
                                            input_shape=(img_height, img_width, 3))
#base_model.trainable = False

#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#predictions = Dense(num_classes, activation='sigmoid')(x)
#model = Model(inputs=base_model.input, outputs=predictions)

#inputs = Input(shape=(img_height, img_width, 3))
#x = base_model(inputs, training=False)
#x = GlobalAveragePooling2D()(x)
#outputs = Dense(num_classes, activation='sigmoid')(x)
#model = Model(inputs, outputs)

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=bp_mll_loss,
              metrics=[bp_mll_loss, utils.euclidean_distance_loss])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_bp_mll_loss', patience=3, restore_best_weights=True)
model.fit(ds_counter, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=math.floor(NUM_IMAGES/BATCH_SIZE), callbacks=[callback], validation_data=(X_test, Y_test))

basepath = os.getcwd()
modelpath = os.path.join(basepath, "../models/classifier")
model.save(modelpath)

preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

for i in range(0, 20):
    y = model.predict(X_test[i, :, :, :].reshape([1, 224, 224, 3]))
    print(y)
    print(Y_test[i, :].reshape([1, 20]))
    print("\n")

model.summary()






































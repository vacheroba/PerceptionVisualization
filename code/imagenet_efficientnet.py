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
import h5py
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

basepath = os.getcwd()
main_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")
encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder.h5")

model = tf.keras.applications.EfficientNetB7(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)

model.summary()

# Load targets (The targets for the decoder are the original inputs, X in main dataset)
hf = h5py.File(main_dataset_path, 'r')
X_test = hf.get('X_Test').value
Y_test = hf.get('Y_Test').value
hf.close()

for i in range(1, 20):
    img = Image.fromarray((X_test[i:i+1, :, :, :]*255).squeeze().astype(np.uint8))
    img = img.resize((600, 600))
    img.show()
    value = np.array(img).reshape([1, 600, 600, 3])
    y = (model.predict(value)).squeeze()
    x = (Y_test[i:i+1, :]).squeeze()
    print(np.argmax(x))
    print(np.argmax(y))
    print("*"*10)
    input("Any key")













































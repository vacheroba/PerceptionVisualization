import pandas as pd
import numpy as np
from PIL import Image
import os
import importdataset
from keras import applications, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from keras import metrics
import keras.backend as K


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


class_names = importdataset.CLASS_NAMES
X_train, X_test, Y_train, Y_test = importdataset.load_dataset()

img_height, img_width = 224, 224
num_classes = 20

base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False,
                                            input_shape=(img_height, img_width, 3))
base_model.trainable = False

#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#predictions = Dense(num_classes, activation='sigmoid')(x)
#model = Model(inputs=base_model.input, outputs=predictions)

inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(),
              loss=euclidean_distance_loss,
              metrics=[metrics.BinaryAccuracy(), euclidean_distance_loss])
model.fit(X_train, Y_train, epochs=2, batch_size=64)

preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

for i in range(0, 20):
    y = model.predict(X_test[i, :, :, :])
    print(y)
    print(Y_test[i, :, :, :])
    print("\n")

model.summary()






































import pandas as pd
import numpy as np
from PIL import Image
import os
import importdataset
from keras import applications
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError

X_train, Y_train, X_test, Y_test, classes = importdataset.load_dataset()

img_height, img_width = 224, 224
num_classes = 20

base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False,
                                            input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='relu')(x)
model = Model(inputs=base_model.input, outputs=predictions)

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=MeanSquaredError(), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=64)

preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

model.summary()





































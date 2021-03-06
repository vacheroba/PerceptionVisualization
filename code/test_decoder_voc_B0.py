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
from PIL import Image
from PIL import ImageTk, ImageWin
import tkinter
import keras
from bpmll import bp_mll_loss
import utils
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names, class_index=-1
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer("resnet50").get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.get_layer("resnet50").input, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    # classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    # x = classifier_input
    # for layer_name in classifier_layer_names:
    #     x = model.get_layer(layer_name)(x)

    # classifier_model = keras.Model(classifier_input, x)
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = model.get_layer("global_average_pooling2d")(x)
    x = model.get_layer("dense")(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array, training=False)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output, training=False)
        if class_index == -1:
            pred_index = tf.argmax(preds[0])
        else:
            pred_index = class_index
        class_channel = preds[:, pred_index]
        print("*"*50)
        print("Taping gradients \nPredictions are")
        print(preds)
        print("I am looking at")
        print(class_channel)
        print("*"*50)

    # This is the gradient of the class score with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    print("Shape of the gradient of the class score wrt last conv layer activations")
    print(grads.shape)
    print("*"*50)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    print("Shape of the pooled gradients (a_k^c)")
    print(pooled_grads.shape)
    print("*" * 50)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for k in range(pooled_grads.shape[-1]):
        # Computes a_k^c * A^k
        last_conv_layer_output[:, :, k] *= pooled_grads[k]

    heatmap = tf.reduce_sum(last_conv_layer_output, axis=2)
    heatmap = tf.nn.relu(heatmap).numpy()

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    #   heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) if np.max(heatmap) > 0.0 else 1.0)
    return heatmap


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

basepath = os.getcwd()
decoder_path = os.path.join(basepath, "../models/decoder_voc_B0")
decoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder_voc_B0.h5")
voc_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")
imagenet_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder_imagenet_rescaled_5k.h5")
classes_file = open(os.path.join(basepath, "../datasets/imagenet_classes.txt"))
imagenet_classes = classes_file.readlines()

decoder = keras.models.load_model(decoder_path, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss})

classifier = tf.keras.applications.EfficientNetB0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)

encoder = keras.Model(classifier.input, classifier.get_layer("top_activation").output)

decoder.summary()
classifier.summary()

# Load targets (The targets for the decoder are the original inputs, X in main dataset)
with h5py.File(decoder_dataset_path, 'r') as hf, h5py.File(voc_dataset_path, 'r') as voc, h5py.File(imagenet_dataset_path, 'r') as imn:
    # X_train = voc.get('X_Test').value  # when using decoder dataset voc b0 'X_Train' or 'X_Test'
    X_train = imn.get('X_train').value  # when using decoder dataset imagenet
    # E_train = hf.get('E_train').valueh

root = tkinter.Tk()
root.geometry('1000x1000')
canvas = tkinter.Canvas(root, width=999, height=999)
canvas.pack()

for i in range(40, 250):
    encoding = encoder.predict(X_train[i:i+1, :, :]*255)
    print("Encoding")
    print(encoding.shape)
    x = ((X_train[i:i+1, :, :])*255).squeeze().astype(np.uint8)
    y = ((decoder.predict(encoding[0:1, :, :, :]))*255).squeeze().astype(np.uint8)  # Times 255 to have a visible image
    res = np.concatenate((x, y), axis=1)
    print("Model prediction")
    classes = classifier.predict(X_train[i:i+1, :, :]*255).squeeze()
    print(classes)
    print(np.argmax(classes))
    print(imagenet_classes[np.argmax(classes)])

    image = Image.fromarray(res)
    image = ImageTk.PhotoImage(image)
    imagesprite = canvas.create_image(400, 800, image=image)
    root.update()
    input("Any key to continue")




































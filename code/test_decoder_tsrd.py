# Takes the standard resnet trained for pascal VOC and retrains the last layer
# Reconstructions are untouched but the network can in this way be used for other tasks
# In this case, the task is that of traffic sign classification from TSRD

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
import tkinter
from PIL import ImageTk, ImageWin
import math
from keras.losses import binary_crossentropy
import keras
import matplotlib.cm as cm

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names, class_index=-1
):
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer("resnet50").get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.get_layer("resnet50").input, last_conv_layer.output)

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        x = model.get_layer("spatial_dropout2d")(x)
        x = model.get_layer("batch_normalization")(x)
        x = model.get_layer("global_average_pooling2d")(x)
        x = model.get_layer("dense")(x)
        classifier_model = keras.Model(classifier_input, x)

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

    # This is the gradient of the class score with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

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

    # We load the original image
    mean = np.mean(img_array.squeeze(), axis=2)
    img = np.zeros([224, 224, 3])
    img[:, :, 0] = mean
    img[:, :, 1] = mean
    img[:, :, 2] = mean

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    # save_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    # save_path = "N:/PycharmProjects/PerceptionVisualization/graphviz/visualizations/cam_4.jpg"
    # save_img.save(save_path)
    # exit()

    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    # superimposed_img = keras.preprocessing.image.img_to_array(superimposed_img)
    var = np.zeros([7, 7, 3])
    var[:, :, 0] = heatmap
    var[:, :, 1] = heatmap
    var[:, :, 2] = heatmap
    heatmap = var
    heatmap = keras.preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    heatmap = keras.preprocessing.image.img_to_array(heatmap)
    heatmap = heatmap.astype(np.float32) / 255.0

    return superimposed_img, heatmap


# ----------------------------------------------------------------------------------------------------------------------
basepath = os.getcwd()
tsrd_path = "N:/PycharmProjects/scratchthat/datasets/tsrd/classes"
classifier_path = os.path.join(basepath, "../models/classifier")
decoder_path = os.path.join(basepath, "../models/decoder_dsim_250_0.4dsim_0.2rec_0.4ssim")

BATCH_SIZE = 64  # 16 for my pc
NUM_EPOCHS = 100
VALID_SPLIT = 0.15

physical_devices = tf.config.list_physical_devices('GPU')

for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)

img_height, img_width = 224, 224
num_classes = 58


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    tsrd_path,
    validation_split=VALID_SPLIT,
    subset="training",
    seed=2222,
    image_size=(img_width, img_height),
    batch_size=BATCH_SIZE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

classifier = keras.models.load_model(classifier_path, custom_objects={"bp_mll_loss": bp_mll_loss,
                                                                 "euclidean_distance_loss": utils.euclidean_distance_loss,
                                                                 "rgb_ssim_loss": utils.rgb_ssim_loss})

decoder = keras.models.load_model(decoder_path, custom_objects={"bp_mll_loss": bp_mll_loss,
                                                                "euclidean_distance_loss": utils.euclidean_distance_loss,
                                                                "rgb_ssim_loss": utils.rgb_ssim_loss})

encoder = keras.Model(classifier.input, classifier.get_layer("global_average_pooling2d").input)

root = tkinter.Tk()
root.geometry('900x800')
canvas = tkinter.Canvas(root, width=896, height=800)
canvas.pack()

for images, labels in dataset:
    for i in range(BATCH_SIZE):
        prediction = np.argmax(np.array(classifier(images[i:i + 1])))
        target = np.array(labels[i])
        reconstruction = decoder(encoder(images[i:i+1]))
        print("Target: " + str(target) + " Pred: " + str(prediction))

        originalimage = (np.array(images[i:i+1]).reshape([224, 224, 3]) * 255).astype(np.uint8)
        reconstructedimage = (np.array(reconstruction).reshape([224, 224, 3]) * 255).astype(np.uint8)

        heatmap, raw_heatmap = make_gradcam_heatmap(
            np.array(images[i:i+1]), classifier, "conv5_block3_out", [], -1
        )

        heatmap = np.array(heatmap)
        white = np.ones([224, 224, 3])  # This is all ones, so a white image
        masked_map = np.multiply(raw_heatmap, np.array(reconstruction).reshape([224, 224, 3])) + np.multiply((white - raw_heatmap), white)
        masked_map = (masked_map.reshape([224, 224, 3])*255).astype(np.uint8)

        image = Image.fromarray(np.concatenate((originalimage, reconstructedimage, masked_map), axis=1))

        photoimage = ImageTk.PhotoImage(image)
        imagesprite = canvas.create_image(0, 0, image=photoimage, anchor="nw")
        root.update()

        # if target != prediction:
        input("Wait")



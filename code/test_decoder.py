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
from PIL import Image, ImageDraw, ImageFont
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
import random

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


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

basepath = os.getcwd()
decoder_path = os.path.join(basepath, "../models/decoder_dsim_250_0.4dsim_0.2rec_0.4ssim")
decoder_ssim_path = os.path.join(basepath, "../models/decoder_ssim")
# decoder_ssim_path = os.path.join(basepath, "../models/decoder_gan_Experiment3(good)")
classifier_path = os.path.join(basepath, "../models/classifier")
main_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")
encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder.h5")

decoder = keras.models.load_model(decoder_path, custom_objects={"bp_mll_loss": bp_mll_loss,
                                                            "euclidean_distance_loss": utils.euclidean_distance_loss,
                                                            "rgb_ssim_loss": utils.rgb_ssim_loss})

decoder_ssim = keras.models.load_model(decoder_ssim_path, custom_objects={"bp_mll_loss": bp_mll_loss,
                                                            "euclidean_distance_loss": utils.euclidean_distance_loss,
                                                            "rgb_ssim_loss": utils.rgb_ssim_loss})

classifier = keras.models.load_model(classifier_path, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss, "rgb_ssim_loss": utils.rgb_ssim_loss})

decoder.summary()
classifier.summary()

# Load targets (The targets for the decoder are the original inputs, X in main dataset)
hf_main = h5py.File(main_dataset_path, 'r')
# Y_train = hf.get('X_Train').value
X_test = hf_main['X_Test']
Y_test = hf_main['Y_Test']

hf_enc = h5py.File(encoder_dataset_path, 'r')
# X_train = hf.get('E_train').value
E_test = hf_enc['E_test']

root = tkinter.Tk()
root.geometry('900x800')
canvas = tkinter.Canvas(root, width=896, height=800)
canvas.pack()

def viz_and_save(idx):
    # infofile = open(os.path.join(basepath, "../images/info.txt"), 'w')
    labelsfile = open(os.path.join(basepath, "../images/labels.txt"), 'w')
    predictionsfile = open(os.path.join(basepath, "../images/predictions.txt"), 'w')
    explainedfile = open(os.path.join(basepath, "../images/explained.txt"), 'w')
    for i in idx:
        reconstructed = ((decoder.predict(E_test[i:i+1, :, :, :]))*255).squeeze().astype(np.uint8)
        reconstructed_ssim = ((decoder_ssim.predict(E_test[i:i+1, :, :, :]))*255).squeeze().astype(np.uint8)
        original = (X_test[i:i+1, :, :, :]*255).squeeze().astype(np.uint8)
        heatmap, raw_heatmap = make_gradcam_heatmap(
            X_test[i:i + 1, :, :, :], classifier, "conv5_block3_out", [], -1
        )

        heatmap = np.array(heatmap)

        res = np.concatenate((original, reconstructed, reconstructed_ssim, heatmap), axis=1)
        t = (heatmap.astype(np.float32)/255.0)*0.3
        multi_map = np.concatenate((t, t, t, t), axis=1)
        multi_map = multi_map + 0.7*(res.astype(np.float32)/255.0)

        # Use colour gamma
        # raw_heatmap = raw_heatmap**(2/3)

        # Threshold
        # threshold = 0.3
        # raw_heatmap[raw_heatmap > threshold] = 1.0
        # raw_heatmap[raw_heatmap <= threshold] = 0.0

        mask = np.concatenate((raw_heatmap, raw_heatmap, raw_heatmap, raw_heatmap), axis=1)  # This is the heatmap scaled 0...1
        masked_map = (res.astype(np.float32)/255.0)  # This is a copy of the full row
        white = np.ones([224, 224*4, 3])  # This is all ones, so a white image

        masked_map = np.multiply(mask, masked_map) + np.multiply((white-mask), white)

        multi_map_int = (multi_map*255.0).astype(np.uint8)
        masked_map = (masked_map*255.0).astype(np.uint8)

        res = np.concatenate((res, multi_map_int, masked_map), axis=0)

        print("\n\n")
        target = Y_test[i, :]

        print("Targets")
        labelline = ""
        predline = ""
        expline = ""

        correct = []
        count = 0
        for elem in importdataset.CLASS_NAMES:
            if target[count] > 0.1:
                correct.append((elem, target[count]))
                labelline += str(elem) + ", "
            count += 1
        print(correct)

        print("Model prediction")
        classes = classifier.predict(X_test[i:i+1, :, :, :]).squeeze()
        # print(classes)
        accepted = []
        count = 0
        for elem in importdataset.CLASS_NAMES:
            if classes[count] > 0.5:
                accepted.append((elem, classes[count]))
                predline += str(elem) + ", "
            count += 1
        argmax = classes.argmax()
        accepted.append(("TOP: "+importdataset.CLASS_NAMES[argmax], classes[argmax]))
        expline = importdataset.CLASS_NAMES[argmax]
        image = Image.fromarray(res)
        photoimage = ImageTk.PhotoImage(image)
        imagesprite = canvas.create_image(0, 0, image=photoimage, anchor="nw")
        root.update()
        print(accepted)
        #sv = input("Any key to continue")
        sv = "ehwjrejrejtejt"

        predline = predline[:-2]+"\n"
        labelline = labelline[:-2] + "\n"
        expline = expline + "\n"

        predictionsfile.writelines(predline)
        labelsfile.writelines(labelline)
        explainedfile.writelines(expline)

        original = res[0:224, 0:224, :]
        cammask = res[448:672, 0:224, :]
        mask_1 = res[448:672, 224:448, :]
        mask_2 = res[448:672, 448:672, :]

        opt1 = Image.fromarray(np.concatenate((original, cammask, mask_1), axis=1))
        opt2 = Image.fromarray(np.concatenate((original, cammask, mask_2), axis=1))

        if sv == "1":
            opt1.save(os.path.join(basepath, "../images/" + str(i) + ".jpg"))

        if sv == "2":
            opt2.save(os.path.join(basepath, "../images/" + str(i) + ".jpg"))

            #original = res[0:224, 0:224, :]
            #masked_original = res[448:, 0:224, :]
            #masked_recon = res[448:, 448:672, :]

            #original_cam = Image.fromarray(np.concatenate((original, masked_original), axis=1))
            #original_viz = Image.fromarray(np.concatenate((original, masked_recon), axis=1))
            #original = Image.fromarray(original)
            #masked_original = Image.fromarray(masked_original)
            #masked_recon = Image.fromarray(masked_recon)

            #original.save(os.path.join(basepath, "../images/original/"+str(i)+".jpg"))
            #masked_original.save(os.path.join(basepath, "../images/cam/"+str(i)+".jpg"))
            #masked_recon.save(os.path.join(basepath, "../images/viz/"+str(i)+".jpg"))
            #original_cam.save(os.path.join(basepath, "../images/original+cam/"+str(i)+".jpg"))
            #original_viz.save(os.path.join(basepath, "../images/original+viz/"+str(i)+".jpg"))
            # infofile.writelines([str(i)+"; "+str(correct)+"; "+str(accepted)+"\n"])
        if sv == "q":
            # infofile.close()
            exit()
    # infofile.close()
    predictionsfile.close()
    explainedfile.close()
    labelsfile.close()


def info_perm():
    infofile = open(os.path.join(basepath, "../images/sorted/info.txt"), 'w')
    rand_perm = np.array(
        [38, 72, 12, 42, 65, 15, 0, 10, 45, 95, 58, 62, 3, 61, 90, 35, 18, 36, 107, 101, 13, 53, 21, 26, 9, 59, 41, 60,
         93, 33])

    for i in rand_perm:
        output_set = set()

        # Add the model's top prediction to the set
        classes = classifier.predict(X_test[i:i + 1, :, :, :]).squeeze()
        argmax = classes.argmax()
        output_set.add(importdataset.CLASS_NAMES[argmax])

        # Merge with actual labels until we fill the 4 cases
        count = 0
        target = Y_test[i, :]
        for elem in importdataset.CLASS_NAMES:
            if target[count] > 0.1 and len(output_set) < 4:
                output_set.add(elem)
            count += 1

        # Add random options if we haven't reached the 4 mark yet
        while len(output_set) < 4:
            ran = random.randrange(0, len(importdataset.CLASS_NAMES))
            output_set.add(importdataset.CLASS_NAMES[ran])

        infofile.writelines(str(output_set)+"\n")

    infofile.close()


def create_additional_viz(idxs):
    participants_cam = 58
    participants_pv = 40

    data_cam = [57, 0, 3, 10, 0, 58, 47, 58, 13, 1, 57, 58, 58, 3, 12, 0, 58, 57, 3, 17, 58, 58, 58, 58, 58, 1, 2, 54, 1, 4]
    data_pv = [33, 10, 36, 30, 3, 32, 34, 29, 13, 13, 29, 27, 25, 17, 31, 1, 38, 32, 17, 31, 19, 38, 28, 41, 28, 4, 15, 30, 5, 7]

    optionsfile = open(os.path.join(basepath, "../images/options.txt"), 'r')
    labelsfile = open(os.path.join(basepath, "../images/labels.txt"), 'r')
    predictionsfile = open(os.path.join(basepath, "../images/predictions.txt"), 'r')
    explainedfile = open(os.path.join(basepath, "../images/explained.txt"), 'r')

    optionlines = optionsfile.readlines()
    labelslines = labelsfile.readlines()
    predictionlines = predictionsfile.readlines()
    explainedlines = explainedfile.readlines()

    font = ImageFont.truetype(os.path.join(basepath, "../Gidole-Regular.ttf"), size=15)

    count = 0
    for idx in idxs:
        labels = labelslines[count]
        predictions = predictionlines[count]
        explained = explainedlines[count].replace('\n', '')

        options = optionlines[count].replace('}', '').replace('{', '').replace('\n', '').replace("'", '')
        options = "Possible answers: " + options + ", 'I just can't tell'  (Correct answer: "+ explained + ")"

        resultimage = Image.new('RGB', (672+75, 224+100), (255, 255, 255, 255))
        vizimage = Image.open(os.path.join(basepath, "../images/"+str(idx)+".jpg"))
        predimage = Image.new('RGB', (224, 75), (255, 255, 255, 255))
        d = ImageDraw.Draw(predimage)
        d.text((10, 10), "Target:\nPredicted:\nExplained:", fill=(0, 0, 0), font=font)
        d.text((100, 10), labels + predictions + explained, fill=(0, 0, 0), font=font)
        predimage = predimage.rotate(270, expand=True)
        optionsimage = Image.new('RGB', (672, 50), (255, 255, 255, 255))
        d = ImageDraw.Draw(optionsimage)
        d.text((10, 10), options, fill=(0, 0, 0), font=font)
        perfimage = Image.new('RGB', (672, 60), (255, 255, 255, 255))
        d = ImageDraw.Draw(perfimage)
        d.text((10, 10), "Grad-CAM users that correctly guessed: ", fill=(0, 0, 0), font=font)
        d.text((10, 35), "PV users that correctly guessed: ", fill=(0, 0, 0), font=font)
        d.text((260, 10), str(data_cam[count])+"/"+str(participants_cam) + " ("+str(round(float(data_cam[count])/float(participants_cam)*float(100), 2))+"%)", fill=(0, 0, 0), font=font)
        d.text((260, 35), str(data_pv[count])+"/"+str(participants_pv) + " ("+str(round(float(data_pv[count])/float(participants_pv)*float(100), 2))+"%)", fill=(0, 0, 0), font=font)

        resultimage.paste(vizimage, (0, 0))
        resultimage.paste(predimage, (672, 0))
        resultimage.paste(optionsimage, (0, 224))
        resultimage.paste(perfimage, (0, 259))

        resultimage.save(os.path.join(basepath, "../additionalmat/"+str(count)+".jpg"))
        count += 1



if __name__ == "__main__":
    rand_perm = np.array(
        [38, 72, 12, 42, 65, 15, 0, 10, 45, 95, 58, 62, 3, 61, 90, 35, 18, 36, 107, 101, 13, 53, 21, 26, 9, 59, 41, 60,
         93, 33])

    #viz_and_save(rand_perm)
    viz_and_save(range(801, 1000))

    # create_additional_viz(rand_perm)






























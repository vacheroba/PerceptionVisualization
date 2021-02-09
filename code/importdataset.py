#Using python 3.7
import os
import numpy as np
import pandas as pd
import PIL
from PIL import Image

CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# Returns a dictionary where for each sample name '2018_0004' there is a vector [0,0,1,0,...] of the target classes
def get_annotations():
    basepath = os.getcwd()
    annotpath = os.path.join(basepath, "../datasets/VOC2012/ImageSets/Main")
    try:
        os.mkdir(os.path.join(basepath, "../datasets/VOC2012/ClassVectors"))
    except:
        pass

    # Takes class-wise annotation from Imagesets/Main and encodes them in a dictionary
    vectordict = dict()
    count = 0
    for cl in CLASS_NAMES:
        file = open(os.path.join(annotpath, cl+"_trainval.txt"), "r")
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            if line == "":
                break
            split = line.split(" ")
            if(split[1] == "-1"):
                vectordict[tuple((split[0], cl))] = 0
            elif(split[2] == "1"):
                vectordict[tuple((split[0], cl))] = 1
            elif(split[2] == "0"):
                vectordict[tuple((split[0], cl))] = 1
                count += 1
        file.close()
    print(count)
    # Merges the dictionary per class, obtaining a dictionary of (image_name, [target])
    targets = dict()
    file = open(os.path.join(annotpath, "aeroplane_trainval.txt"), "r")
    lines = file.readlines()
    for line in lines:
        vector = []
        line = line.strip('\n')
        if line == "":
            break
        split = line.split(" ")
        for cl in CLASS_NAMES:
            vector.append(vectordict[tuple((split[0], cl))])
        targets[split[0]] = vector
    file.close()

    return targets


# Returns the dataset
def load_dataset():
    basepath = os.getcwd()
    imagespath = os.path.join(basepath, "../datasets/VOC2012/JPEGImages")

    targets: dict = get_annotations()
    num_images = len(targets)
    splitidx = int(num_images*0.7)

    images = np.array([224, 224, 3, num_images])
    targets = np.array([len(CLASS_NAMES), num_images])

    counter = 0
    while True:
        try:
            item = targets.popitem()
        except:
            break

        image: Image = Image.open(os.path.join(imagespath, item[0])).resize((224, 224))
        images[:, :, :, counter] = np.array(image)
        targets[:, counter] = np.array(item[1])

    return images[:, :, :, 0:splitidx], targets[:, 0:splitidx], \
        images[:, :, :, splitidx:], targets[:, splitidx:], \
        CLASS_NAMES























































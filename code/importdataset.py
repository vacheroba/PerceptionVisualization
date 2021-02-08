#Using python 3.7
import os


CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

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

print(targets)

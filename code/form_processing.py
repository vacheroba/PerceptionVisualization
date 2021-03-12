import numpy as np
from shutil import copyfile
import os

correct = np.array([0, 3, 9, 10, 12, 13, 15, 18, 21, 26, 38, 53, 58, 60, 62])
wrong = np.array([33, 35, 36, 41, 42, 45, 59, 61, 65, 72, 90, 93, 95, 101, 107])


def save_random_perm():
    idx = np.concatenate((correct, wrong))
    idx = np.random.permutation(idx)
    print(idx)

    basepath = os.getcwd()

    count = 1
    for i in idx:
        copyfile(os.path.join(basepath, "../images/original/"+str(i)+".jpg"), os.path.join(basepath, "../images/sorted/original/"+str(count)+".jpg"))
        copyfile(os.path.join(basepath, "../images/cam/"+str(i)+".jpg"), os.path.join(basepath, "../images/sorted/cam/"+str(count)+".jpg"))
        copyfile(os.path.join(basepath, "../images/viz/"+str(i)+".jpg"), os.path.join(basepath, "../images/sorted/viz/"+str(count)+".jpg"))
        copyfile(os.path.join(basepath, "../images/original+cam/"+str(i)+".jpg"), os.path.join(basepath, "../images/sorted/original+cam/"+str(count)+".jpg"))
        copyfile(os.path.join(basepath, "../images/original+viz/"+str(i)+".jpg"), os.path.join(basepath, "../images/sorted/original+viz/"+str(count)+".jpg"))
        count += 1




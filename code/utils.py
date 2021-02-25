import keras.backend as K
import tensorflow as tf

SSIM_GAMMA = 11000.0


def euclidean_distance_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)


def euclidean_ssim_loss(y_true, y_pred):
    # return K.sum(K.square(y_pred - y_true), axis=(1, 2, 3))
    # return SSIM_GAMMA*(tf.image.ssim(y_pred, y_true, max_val=1.0))
    return K.sum(K.square(y_pred - y_true), axis=(1, 2, 3)) - SSIM_GAMMA*(tf.image.ssim(y_pred, y_true, max_val=1.0))


def squared_means_loss(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


def rgb_ssim_loss(y_true, y_pred):
    # return K.sum(K.square(y_pred - y_true), axis=(1, 2, 3))
    # return SSIM_GAMMA*(tf.image.ssim(y_pred, y_true, max_val=1.0))
    return -(tf.image.ssim(y_pred[:, :, :, 0:1], y_true[:, :, :, 0:1], max_val=1.0) +
             tf.image.ssim(y_pred[:, :, :, 1:2], y_true[:, :, :, 1:2], max_val=1.0) +
             tf.image.ssim(y_pred[:, :, :, 2:3], y_true[:, :, :, 2:3], max_val=1.0))

if __name__ == "__main__":
    import numpy as np
    import os
    import h5py
    basepath = os.getcwd()
    encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder_imagenet_rescaled.h5")
    voc_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")

    hf = h5py.File(encoder_dataset_path, 'r')
    im1 = hf["X_train"][1:2, :, :, :]
    im1 = np.repeat(im1, 10, axis=0)
    im2 = hf["X_train"][2:3, :, :, :]
    im2 = np.repeat(im2, 10, axis=0)
    print(np.array(rgb_ssim_loss(im1, im2)))

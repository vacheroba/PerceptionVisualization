import keras.backend as K
import tensorflow as tf
import keras.layers as layers
import keras.losses as losses

SSIM_GAMMA = 11000.0

cross_entropy = tf.keras.losses.BinaryCrossentropy()


def euclidean_distance_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=(1, 2, 3))  # axis = -1


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
    return - (tf.image.ssim(y_pred[:, :, :, 0:1], y_true[:, :, :, 0:1], max_val=1.0) +
              tf.image.ssim(y_pred[:, :, :, 1:2], y_true[:, :, :, 1:2], max_val=1.0) +
              tf.image.ssim(y_pred[:, :, :, 2:3], y_true[:, :, :, 2:3], max_val=1.0))


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3, strides=2, padding='same', input_shape=[224, 224, 3]))
    model.add(layers.Conv2D(64, 3, strides=1, padding='same', input_shape=[224, 224, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, 3, strides=2, padding='same'))
    model.add(layers.Conv2D(128, 3, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, 3, strides=2, padding='same'))
    model.add(layers.Conv2D(256, 3, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, 3, strides=2, padding='same'))
    model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(1024, 3, strides=2, padding='same'))
    model.add(layers.Conv2D(1024, 3, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def make_decoder_model():
    model = tf.keras.Sequential()
    # First block (pool->conv->conv->conv)
    model.add(layers.Conv2DTranspose(input_shape=(7, 7, 2048), filters=512, kernel_size=2, padding="same", strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
    # Second block (pool->conv->conv->conv)
    model.add(layers.Conv2DTranspose(filters=512, kernel_size=2, padding="same", strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=512, padding="same", kernel_size=3, activation="relu"))
    # Third block (pool->conv->conv->conv)
    model.add(layers.Conv2DTranspose(filters=256, kernel_size=2, padding="same", strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=256, padding="same", kernel_size=3, activation="relu"))
    # Fourth block (pool->conv->conv)
    model.add(layers.Conv2DTranspose(filters=128, kernel_size=2, padding="same", strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=128, padding="same", kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=128, padding="same", kernel_size=3, activation="relu"))
    # Fifth block (pool->conv->conv)
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=2, padding="same", strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, padding="same", kernel_size=3, activation="relu"))
    model.add(layers.Conv2D(filters=64, padding="same", kernel_size=3, activation="relu"))
    # Output (conv)
    model.add(layers.Conv2D(filters=3, padding="same", kernel_size=3, activation="sigmoid"))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output)*0.95, real_output)
    fake_loss = cross_entropy(tf.ones_like(real_output)*0.05, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def test_losses():
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
    im1 = tf.convert_to_tensor(im1)
    im2 = tf.convert_to_tensor(im2)
    y2 = tf.constant(3.0)
    with tf.GradientTape() as g:
        g.watch(y2)
        g.watch(im1)
        g.watch(im2)
        y2 = rgb_ssim_loss(im1, im2)
    print(np.max(np.array(g.gradient(y2, im1))))


def test_discriminator():
    model = make_discriminator_model()
    model.summary()


if __name__ == "__main__":
    test_discriminator()

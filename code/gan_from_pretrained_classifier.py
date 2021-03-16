import pandas as pd
import numpy as np
from PIL import Image
import os
import importdataset
from keras import applications, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Flatten, Conv2DTranspose, BatchNormalization, SpatialDropout2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from keras import metrics
from keras import losses
from keras.models import Sequential
import keras.backend as K
import keras
from bpmll import bp_mll_loss
import utils
import h5py
import tensorflow as tf
import random
import math
import wandb

# ----------------------------------------------------------------------------------------------------------WANDB PARAMS
wandb.init(project='PerceptionVisualization', entity='loris2222')
wandbconfig = wandb.config

BATCH_SIZE = 64
BUFFER_SIZE = 10
EPOCHS = 200

LEARN_RATE_DEC = 1e-4
LEARN_RATE_DISC = 2*1e-5
BETA1_DISC = 0.5

START_PRETRAINED = False

WEIGHT_GAN_LOSS = 1.0
WEIGHT_REC_LOSS = 1.0
WEIGHT_DSIM_LOSS = 1.0

TRAIN_DISC_LOWER_THRESH = 0.0
TRAIN_DEC_UPPER_THRESH = 2.0

DISC_MODEL = "(conv stride 2>conv stride 1>batchnorm)*4>globAvPool>sigmoid  filters 64 64 128 128 256 256 512 512\n" \
             "discriminator and generator losses have now tf.math.log added to them"

wandbconfig.update({"batch_size": BATCH_SIZE, "buffer_size": BUFFER_SIZE,
                    "epochs": EPOCHS, "learn_rate_dec":LEARN_RATE_DEC,
                    "learn_rate_disc": LEARN_RATE_DISC, "beta1_disc": BETA1_DISC,
                    "start_pretrained": START_PRETRAINED, "weight_gan": WEIGHT_GAN_LOSS,
                    "weight_rec": WEIGHT_REC_LOSS, "weight_dsim": WEIGHT_DSIM_LOSS,
                    "disc_thresh": TRAIN_DISC_LOWER_THRESH, "dec_thresh": TRAIN_DEC_UPPER_THRESH,
                    "disc_model_info": DISC_MODEL})
# ----------------------------------------------------------------------------------------------------------------------

physical_devices = tf.config.list_physical_devices('GPU')

for h in physical_devices:
    tf.config.experimental.set_memory_growth(h, True)

basepath = os.getcwd()
main_dataset_path = os.path.join(basepath, "../datasets/dataset.h5")
encoder_dataset_path = os.path.join(basepath, "../datasets/dataset_encoder.h5")

decoderpath = os.path.join(basepath, "../models/decoder")
classifierpath = os.path.join(basepath, "../models/classifier")

with h5py.File(encoder_dataset_path, 'r') as enc:
    NUM_IMAGES = enc["E_train"].shape[0]
    print("Dataset info")
    print(enc["E_train"].shape)
    BATCH_COUNT = 0
    for i in range(0, NUM_IMAGES - BATCH_SIZE, BATCH_SIZE):
        BATCH_COUNT += 1

bynary_permute = np.concatenate((np.repeat(np.array([0]), math.floor(BATCH_SIZE/2)), np.repeat(np.array([1]), math.floor(BATCH_SIZE/2))))

def generator():
    with h5py.File(main_dataset_path, 'r') as ds, h5py.File(encoder_dataset_path, 'r') as enc:
        for i in range(0, NUM_IMAGES-BATCH_SIZE, BATCH_SIZE):
            real_images = None
            fake_images = None
            embeddings = None

            # Compile list of real and fake idxs
            permutation = np.random.permutation(bynary_permute)
            for j in range(0, BATCH_SIZE):
                rand = permutation[j]
                if rand == 1:
                    if real_images is None:
                        real_images = np.array(ds["X_Train"][i+j:i+j+1, :, :, :])
                    else:
                        real_images = np.append(real_images, ds["X_Train"][i+j:i+j+1, :, :, :], axis=0)
                else:
                    if fake_images is None:
                        fake_images = np.array(ds["X_Train"][i+j:i+j+1, :, :, :])
                        embeddings = np.array(enc["E_train"][i+j:i+j+1, :, :, :])
                    else:
                        fake_images = np.append(fake_images, ds["X_Train"][i+j:i+j+1, :, :, :], axis=0)
                        embeddings = np.append(embeddings, enc["E_train"][i+j:i+j+1, :, :, :], axis=0)

            yield tf.convert_to_tensor(real_images, dtype=tf.float32), tf.convert_to_tensor(fake_images, dtype=tf.float32), tf.convert_to_tensor(embeddings, dtype=tf.float32)

# Load targets (The targets for the decoder are the original inputs, X in main dataset)
hf = h5py.File(main_dataset_path, 'r')
Y_test = hf.get('X_Test').value

# Load inputs(The outputs of the encoder, E in encoder dataset)
hf.close()
hf = h5py.File(encoder_dataset_path, 'r')
E_test = hf.get('E_test').value
hf.close()

if START_PRETRAINED:
    decoder = keras.models.load_model(decoderpath, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss})
else:
    decoder = utils.make_decoder_model()

classifier = keras.models.load_model(classifierpath, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss})
encoder = keras.Model(classifier.input, classifier.get_layer("global_average_pooling2d").input)
discriminator = utils.make_discriminator_model()

decoder_optimizer = tf.keras.optimizers.Adam(LEARN_RATE_DEC)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE_DISC, beta_1=BETA1_DISC)


def deep_sim_loss(images, y_pred):
    y_true = encoder(images, training=False)
    return K.sum(K.square(y_pred - y_true), axis=(1, 2, 3))


@tf.function
def train_step(batch):
    real_images = batch[0][0, :, :, :, :]
    fake_images = batch[1][0, :, :, :, :]
    fake_embeddings = batch[2][0, :, :, :, :]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = decoder(fake_embeddings, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        dec_loss = tf.constant(WEIGHT_GAN_LOSS)*utils.generator_loss(fake_output) + tf.constant(WEIGHT_REC_LOSS)*tf.norm(utils.euclidean_distance_loss(fake_images, generated_images)) + tf.constant(WEIGHT_DSIM_LOSS)*tf.norm(deep_sim_loss(generated_images, fake_embeddings))
        disc_loss = utils.discriminator_loss(real_output, fake_output)

    gradients_of_decoder = gen_tape.gradient(dec_loss, decoder.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    if tf.math.less_equal(disc_loss, tf.constant(TRAIN_DEC_UPPER_THRESH)):
        decoder_optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))

    # Train discriminator only if its loss is greater than value (previously 0.35)
    if tf.math.greater(disc_loss, tf.constant(TRAIN_DISC_LOWER_THRESH)):
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


for epoch in range(EPOCHS):
    ds_counter = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32), (
                                                tf.TensorShape([math.floor(BATCH_SIZE / 2), 224, 224, 3]),
                                                tf.TensorShape([math.floor(BATCH_SIZE / 2), 224, 224, 3]),
                                                tf.TensorShape([math.floor(BATCH_SIZE / 2), 7, 7, 2048])))
    ds_counter = ds_counter.shuffle(BUFFER_SIZE).batch(1)

    print("EPOCH "+str(epoch)+"/"+str(EPOCHS))
    step_counter = 0
    for image_batch in ds_counter:
        print("\rstep "+str(step_counter)+"/"+str(BATCH_COUNT), end='')
        train_step(image_batch)
        step_counter += 1
    print("\n")

    generated_test_images = decoder(E_test[0:100, :, :, :], training=False)
    real_test_output = discriminator(Y_test[0:100, :, :, :], training=False)
    fake_test_output = discriminator(generated_test_images, training=False)

    dec_loss = np.mean(utils.euclidean_distance_loss(generated_test_images, Y_test[0:100, :, :, :]))
    disc_loss = utils.discriminator_loss(real_test_output, fake_test_output)

    wandb.log({"epoch": epoch})
    print("Reconstruction loss " + str(dec_loss))
    wandb.log({"reconstruction loss": dec_loss})
    print("Discriminator loss " + str(disc_loss))
    wandb.log({"discriminator loss": disc_loss})

decoder.save(os.path.join(basepath, "../models/decoder_gan"))
discriminator.save(os.path.join(basepath, "../models/discriminator_gan"))



























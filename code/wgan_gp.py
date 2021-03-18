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

TEST_CONFIG = False

# ----------------------------------------------------------------------------------------------------------WANDB PARAMS
if TEST_CONFIG:
    BATCH_SIZE = 4
else:
    BATCH_SIZE = 32
BUFFER_SIZE = 10
EPOCHS = 20

LEARN_RATE_DEC = 2e-4
LEARN_RATE_DISC = 2e-4
BETA1_DEC = 0.5
BETA1_DISC = 0.5
BETA2_DEC = 0.9
BETA2_DISC = 0.9

START_PRETRAINED = False

WEIGHT_GAN_LOSS = 1.0
WEIGHT_REC_LOSS = 1.0
WEIGHT_DSIM_LOSS = 1.0

WEIGHT_GP = 10.0
DISC_STEPS = 1

GPU_ID = 0

TRAIN_DISC_LOWER_THRESH = 0.01  # minimum 0.0
TRAIN_DEC_UPPER_THRESH = 0.2  # maximum 2.0

DISC_MODEL = "(conv stride 2>conv stride 1>batchnorm)*4>globAvPool>sigmoid  filters 64 64 128 128 256 256 512 512\n" \
             "always training both disc and dec"

if not TEST_CONFIG:
    wandb.init(project='PerceptionVisualization', entity='loris2222')
    wandbconfig = wandb.config
    wandbconfig.update({"batch_size": BATCH_SIZE, "buffer_size": BUFFER_SIZE,
                        "epochs": EPOCHS, "learn_rate_dec":LEARN_RATE_DEC,
                        "learn_rate_disc": LEARN_RATE_DISC, "beta1_dec": BETA1_DEC,
                        "beta2_dec": BETA2_DEC, "beta1_disc": BETA1_DISC, "beta2_disc": BETA2_DISC,
                        "start_pretrained": START_PRETRAINED, "weight_gan": WEIGHT_GAN_LOSS,
                        "weight_rec": WEIGHT_REC_LOSS, "weight_dsim": WEIGHT_DSIM_LOSS,
                        "disc_thresh": TRAIN_DISC_LOWER_THRESH, "dec_thresh": TRAIN_DEC_UPPER_THRESH, "gpu": GPU_ID,
                        "gp_weight": WEIGHT_GP, "discriminator_steps": DISC_STEPS, "disc_model_info": DISC_MODEL})
# ----------------------------------------------------------------------------------------------------------------------

if not TEST_CONFIG:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)  # Use Titan XP on monkey, set to 0 in piggy
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


ds_counter = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32), (
                                                tf.TensorShape([math.floor(BATCH_SIZE / 2), 224, 224, 3]),
                                                tf.TensorShape([math.floor(BATCH_SIZE / 2), 224, 224, 3]),
                                                tf.TensorShape([math.floor(BATCH_SIZE / 2), 7, 7, 2048])))

ds_counter = ds_counter.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
ds_counter = ds_counter.repeat(EPOCHS)

# Load targets (The targets for the decoder are the original inputs, X in main dataset)
hf = h5py.File(main_dataset_path, 'r')
if not TEST_CONFIG:
    Y_test = hf.get('X_Test').value

# Load inputs(The outputs of the encoder, E in encoder dataset)
hf.close()
hf = h5py.File(encoder_dataset_path, 'r')
if not TEST_CONFIG:
    E_test = hf.get('E_test').value
hf.close()

if START_PRETRAINED:
    decoder = keras.models.load_model(decoderpath, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss})
else:
    decoder = utils.make_decoder_model()

classifier = keras.models.load_model(classifierpath, custom_objects={"bp_mll_loss": bp_mll_loss, "euclidean_distance_loss": utils.euclidean_distance_loss})
encoder = keras.Model(classifier.input, classifier.get_layer("global_average_pooling2d").input)
discriminator = utils.make_discriminator_model()

decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE_DEC, beta_1=BETA1_DEC)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE_DISC, beta_1=BETA1_DISC)


def deep_sim_loss(images, y_pred):
    y_true = encoder(images, training=False)
    return tf.reduce_mean(K.square(y_pred - y_true))


def wgan_discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def wgan_generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


class WGAN(keras.Model):
    def __init__(self, enc, disc, dec, latent_dim, discriminator_extra_steps=3, gp_weight=10.0):
        super(WGAN, self).__init__()
        self.encoder = enc
        self.discriminator = disc
        self.generator = dec
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.batch_size = BATCH_SIZE

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([math.floor(batch_size / 2), 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, batch):
        # Batching adds a dimension but my batch is already made up of all images
        real_images = batch[0]
        fake_images_reconstruction_targets = batch[1]
        fake_embeddings = batch[2]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for s in range(self.d_steps):
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(fake_embeddings, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(self.batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(fake_embeddings, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = tf.constant(WEIGHT_GAN_LOSS)*self.g_loss_fn(gen_img_logits)
                     # + tf.constant(WEIGHT_REC_LOSS)*utils.euclidean_distance_loss(fake_images_reconstruction_targets, generated_images)
                     # + tf.constant(WEIGHT_DSIM_LOSS)*deep_sim_loss(generated_images, fake_embeddings)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return {"d_loss": d_loss, "g_loss": g_loss}


class callbacklog(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        if not TEST_CONFIG:
            keys = list(logs.keys())
            print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
            wandb.log({"discriminator loss": logs.keys["d_loss"]})
            wandb.log({"decoder loss": logs.keys["g_loss"]})

# Instantiate the WGAN model.
wgan = WGAN(
    enc=classifier,
    disc=discriminator,
    dec=decoder,
    latent_dim=[7, 7, 2048],
    discriminator_extra_steps=DISC_STEPS,
    gp_weight=WEIGHT_GP
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=decoder_optimizer,
    g_loss_fn=wgan_generator_loss,
    d_loss_fn=wgan_discriminator_loss,
)

# Start training the model.
wgan.fit(ds_counter, batch_size=BATCH_SIZE, epochs=EPOCHS, steps_per_epoch=math.floor(NUM_IMAGES/BATCH_SIZE), callbacks=[callbacklog()])

decoder.save(os.path.join(basepath, "../models/decoder_wgan_gp"))
discriminator.save(os.path.join(basepath, "../models/discriminator_wgan_gp"))



























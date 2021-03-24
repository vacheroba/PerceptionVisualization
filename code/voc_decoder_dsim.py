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
import socket

TEST_CONFIG = False

# ----------------------------------------------------------------------------------------------------------WANDB PARAMS
if TEST_CONFIG:
    BATCH_SIZE = 4  # 4
else:
    BATCH_SIZE = 32  # 64
BUFFER_SIZE = 10
EPOCHS = 500

LEARN_RATE_DEC = 1e-4
LEARN_RATE_DISC = 1e-4
BETA1_DEC = 0.9
BETA1_DISC = 0.9
BETA2_DEC = 0.999
BETA2_DISC = 0.999

START_PRETRAINED = True

WEIGHT_REC_LOSS = 0.5
WEIGHT_DSIM_LOSS = 0.5

host = socket.gethostname()
if TEST_CONFIG or host == "piggypiggy":
    GPU_ID = 0
else:
    GPU_ID = 1

DISC_MODEL = "DECODER DSIM"

if not TEST_CONFIG:
    wandb.init(project='PerceptionVisualization', entity='loris2222')
    wandbconfig = wandb.config
    wandbconfig.update({"batch_size": BATCH_SIZE, "buffer_size": BUFFER_SIZE,
                        "epochs": EPOCHS, "learn_rate_dec":LEARN_RATE_DEC,
                        "learn_rate_disc": LEARN_RATE_DISC, "beta1_dec": BETA1_DEC,
                        "beta2_dec": BETA2_DEC, "beta1_disc": BETA1_DISC, "beta2_disc": BETA2_DISC,
                        "start_pretrained": START_PRETRAINED,
                        "weight_rec": WEIGHT_REC_LOSS, "weight_dsim": WEIGHT_DSIM_LOSS,
                        "gpu": GPU_ID,
                        "disc_model_info": DISC_MODEL})
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

def generator():
    with h5py.File(main_dataset_path, 'r') as ds, h5py.File(encoder_dataset_path, 'r') as enc:
        for i in range(0, NUM_IMAGES-BATCH_SIZE, BATCH_SIZE):
            yield tf.convert_to_tensor(enc["E_train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32), tf.convert_to_tensor(ds["X_Train"][i:i + BATCH_SIZE, :, :, :], dtype=tf.float32)


ds_counter = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32), (tf.TensorShape([BATCH_SIZE, 7, 7, 2048]), tf.TensorShape([BATCH_SIZE, 224, 224, 3])))

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

del classifier

decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE_DEC, beta_1=BETA1_DEC, beta_2=BETA2_DEC)

def deep_sim_loss(images, y_pred):
    y_true = encoder(images, training=False)
    return tf.reduce_mean(K.square(y_pred - y_true))


class DSIM_MODEL(keras.Model):
    def __init__(self, enc, dec, latent_dim):
        super(DSIM_MODEL, self).__init__()
        self.encoder = enc
        self.generator = dec
        self.latent_dim = latent_dim
        self.batch_size = BATCH_SIZE

    def compile(self, g_optimizer):
        super(DSIM_MODEL, self).compile()
        self.g_optimizer = g_optimizer

    def train_step(self, batch):
        # Batching adds a dimension but my batch is already made up of all images
        images = batch[1]
        embeddings = batch[0]

        # Train the generator
        # Get the latent vector
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(embeddings[0:BATCH_SIZE, :, :, :], training=True)
            # Calculate the generator loss
            g_loss = tf.constant(WEIGHT_REC_LOSS)*utils.euclidean_distance_loss(images[0:BATCH_SIZE, :, :, :], generated_images) \
                     + tf.constant(WEIGHT_DSIM_LOSS)*deep_sim_loss(generated_images, embeddings[0:BATCH_SIZE, :, :, :])

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return {"g_loss": g_loss}


# Instantiate the DSIM model.
dsim_model = DSIM_MODEL(
    enc=encoder,
    dec=decoder,
    latent_dim=[7, 7, 2048]
)

# Compile the DSIM model.
dsim_model.compile(
    g_optimizer=decoder_optimizer
)

# Start training the model.
callback = tf.keras.callbacks.EarlyStopping(monitor='g_loss', patience=100, restore_best_weights=False)
dsim_model.fit(ds_counter, batch_size=1, epochs=EPOCHS, steps_per_epoch=math.floor(NUM_IMAGES/BATCH_SIZE), callbacks=[callback])

decoder.save(os.path.join(basepath, "../models/decoder_dsim"))



























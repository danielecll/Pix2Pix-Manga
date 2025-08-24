import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from discord import Webhook, RequestsWebhookAdapter, File

from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Conv2D, Conv2DTranspose, Flatten,
                          BatchNormalization, LeakyReLU, ReLU, Dropout,
                          Concatenate, ZeroPadding2D)
from keras.optimizer_v2.adam import Adam
from keras.losses import BinaryCrossentropy

# Install discord if needed
os.system('pip install discord')

# Create necessary directories
os.makedirs("output/", exist_ok=True)
os.makedirs("weights/gen/", exist_ok=True)
os.makedirs("weights/disc/", exist_ok=True)

# Hyperparameters
BATCH_SIZE = 8
IMAGE_SIZE = 256
LAMBDA = 100
EPOCHS = 5000

# Loss and optimizers
loss_function = BinaryCrossentropy(from_logits=True)
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

# -------------------- Data loading -------------------- #
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    w = tf.shape(image)[1] // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

def resize(input_image, real_image):
    input_image = tf.image.resize(input_image, [IMAGE_SIZE, IMAGE_SIZE],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [IMAGE_SIZE, IMAGE_SIZE],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_jitter(input_image, real_image):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image

def load_train_images(image_path):
    input_image, real_image = load(image_path)
    input_image, real_image = resize(input_image, real_image)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

def load_test_image(image_path):
    input_image, real_image = load(image_path)
    input_image, real_image = resize(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

# -------------------- Model blocks -------------------- #
def downsample(filters, size, batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    result = Sequential()
    result.add(Conv2D(filters, size, strides=2, padding="same",
                      kernel_initializer=init, use_bias=False))
    if batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU())
    return result

def upsample(filters, size, dropout=False):
    init = tf.random_normal_initializer(0., 0.02)
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding="same",
                               kernel_initializer=init, use_bias=False))
    result.add(BatchNormalization())
    if dropout:
        result.add(Dropout(0.5))
    result.add(ReLU())
    return result

# -------------------- Generator -------------------- #
def generator():
    inputs = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
    
    # Downsampling
    down_stack = [
        downsample(64, 4, batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),  
    ]
    
    # Upsampling
    up_stack = [
        upsample(512, 4, dropout=True),  
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    
    init = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(3, 4, strides=2, padding="same",
                           kernel_initializer=init, activation="tanh")
    
    x = inputs
    skips = []
    
    # Downsampling path
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    
    # Upsampling path with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])
    
    x = last(x)
    return Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_function(tf.ones_like(disc_generated_output),
                             disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# -------------------- Discriminator -------------------- #
def discriminator():
    init = tf.random_normal_initializer(0., 0.02)
    inp = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    tar = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="target_image")
    x = Concatenate()([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = ZeroPadding2D()(down3)
    conv = Conv2D(256, 4, strides=1, kernel_initializer=init, use_bias=False)(zero_pad1)
    leaky_relu = LeakyReLU()(conv)
    zero_pad2 = ZeroPadding2D()(leaky_relu)
    last = Conv2D(1, 4, strides=1, kernel_initializer=init)(zero_pad2)
    return Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_function(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_function(tf.zeros_like(disc_generated_output),
                                   disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# -------------------- Training utilities -------------------- #
def save_images(model, test_input, target, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Prediction Image"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.savefig(f"output/epoch_{epoch}.jpg")
    plt.close()
    with open(f'output/epoch_{epoch}.jpg', 'rb') as f:
        webhook = Webhook.from_url(
            'WEBHOOK_URL',
            adapter=RequestsWebhookAdapter()
        )
        webhook.send(f"Epoch: {epoch}", file=File(f))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = gen(input_image, training=True)
        disc_real_output = disc([input_image, target], training=True)
        disc_generated_output = disc([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, gen.trainable_variables))
    print("Updating Generator...")
    if disc_loss > 0.5:
        discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
        print("Updating Discriminator...")
    return gen_total_loss, disc_loss

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        for input_, target in test_ds.take(1):
            save_images(gen, input_, target, epoch)
        print(f"Epoch {epoch}")
        for n, (input_, target) in tqdm(train_ds.enumerate()):
            gen_loss, disc_loss = train_step(input_, target, epoch)
        print(f"Generator loss {gen_loss:.2f} Discriminator loss {disc_loss:.2f}")
        print(f"Time for epoch {epoch+1}: {time.time() - start} sec\n")
        gen.save_weights("weights/gen/gen_checkpoint")
        disc.save_weights("weights/disc/disc_checkpoint")
        webhook = Webhook.from_url(
            'WEBHOOK_URL',
            adapter=RequestsWebhookAdapter()
        )
        webhook.send(content=f"Epoch: {epoch+1}\nTime Taken: {time.time() - start}\n"
                             f"Generator Loss: {gen_loss}\nDiscriminator Loss: {disc_loss}\n⠀")

# -------------------- Initialize models -------------------- #
gen = generator()
gen.load_weights("../input/model-weights/gen/gen_checkpoint")
disc = discriminator()
disc.load_weights("../input/model-weights/disc/disc_checkpoint")

# -------------------- Load datasets -------------------- #
train_dataset = tf.data.Dataset.list_files("../input/roombav2/train/*.jpg")
train_dataset = train_dataset.map(load_train_images)
train_dataset = train_dataset.shuffle(10).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files("../input/roombav2/val/*.jpg")
test_dataset = test_dataset.map(load_test_image)
test_dataset = test_dataset.batch(BATCH_SIZE)

tf.keras.backend.clear_session()
fit(train_dataset, EPOCHS, test_dataset)

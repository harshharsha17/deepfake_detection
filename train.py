# gan_train.py - Clean GAN Training Script
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
LATENT_DIM = 100  # Noise vector size

# Set paths
base_path = 'deepfake_detection'
checkpoint_path = os.path.join(base_path, 'gan_checkpoints')
results_path = os.path.join(base_path, 'gan_results')

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# --- Generator Model ---
def build_generator():
    model = Sequential(name="Generator")
    model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((16, 16, 256)))
    
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    return model

# --- Discriminator Model ---
def build_discriminator():
    model = Sequential(name="Discriminator")
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[IMG_SIZE, IMG_SIZE, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# --- Data loader ---
def create_real_data_loader():
    datagen = ImageDataGenerator(rescale=1./255)

    real_flow = datagen.flow_from_directory(
        'deepfake_detection/data/real_and_fake_face',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=['training_real'],
        class_mode=None,
        shuffle=True
    )
    return real_flow

# --- GAN Trainer ---
class GAN(tf.keras.Model):
    def _init_(self, generator, discriminator):
        super(GAN, self)._init_()
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = BinaryCrossentropy(from_logits=False)

    def compile(self, g_optimizer, d_optimizer):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, LATENT_DIM))

        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([real_images, generated_images], axis=0)

        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))  # Label smoothing

        # Train Discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.cross_entropy(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Train Generator
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(fake_images)
            g_loss = self.cross_entropy(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# --- Main ---
def main():
    print("Preparing real images loader...")
    real_flow = create_real_data_loader()

    print("Building Generator and Discriminator...")
    generator = build_generator()
    discriminator = build_discriminator()

    gan = GAN(generator, discriminator)
    gan.compile(
        g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
    )

    print("Starting GAN training...")
    start_time = time.time()

    gan.fit(
        real_flow,
        epochs=EPOCHS,
        steps_per_epoch=real_flow.samples // BATCH_SIZE
    )

    training_time = time.time() - start_time
    print(f"\nGAN training completed in {training_time:.2f} seconds")

    # Save Generator and Discriminator separately
    generator.save(os.path.join(checkpoint_path, "generator.h5"))
    discriminator.save(os.path.join(checkpoint_path, "discriminator.h5"))
    print(f"Saved generator and discriminator models to {checkpoint_path}")

    # --- Create and Save Final Deepfake Detector Model ---
    print("Creating and saving final deepfake detector model...")
    noise_input = layers.Input(shape=(LATENT_DIM,), name="Noise_Input")
    generated_image = generator(noise_input)
    validity = discriminator(generated_image)
    
    final_model = Model(inputs=noise_input, outputs=validity, name="Final_Deepfake_Detector")
    final_model.save(os.path.join(checkpoint_path, "final_deepfake_detector.h5"))
    print(f"Saved final deepfake detector model to {checkpoint_path}")

if _name_ == "_main_":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU available. Using GPU.")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found. Using CPU.")

    main()
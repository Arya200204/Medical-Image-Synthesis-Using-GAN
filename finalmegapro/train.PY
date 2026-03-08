import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Embedding, Concatenate,
    Conv2DTranspose, LeakyReLU, Dropout, Conv2D
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow_addons as tfa

# === Constants ===
IMG_SIZE = (128, 128)
LATENT_DIM = 100
DATASET_PATH = "processed_dataset"
OUTPUT_GEN = "static/trained_images/generated"
MODEL_SAVE_PATH = "generator_cgan_sn.h5"


# === Load and preprocess images ===
def load_images_labels(dataset_path):
    images, labels = [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    img = load_img(path, color_mode="grayscale", target_size=IMG_SIZE)
                    img = img_to_array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
                    images.append(img)

                    # Use parent folders as labels
                    label = "_".join(path.split(os.sep)[-3:-1])
                    labels.append(label)
                except Exception as e:
                    print(f"❌ Skipped file {file}: {e}")
    return np.array(images), np.array(labels)


# === Generator ===
def build_generator(latent_dim, n_classes):
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(n_classes, latent_dim)(label)
    label_embedding = Flatten()(label_embedding)

    merged = Concatenate()([noise, label_embedding])
    x = Dense(256 * 16 * 16, activation="relu")(merged)
    x = Reshape((16, 16, 256))(x)

    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')(x)

    return Model([noise, label], x)


# === Discriminator ===
def build_discriminator(n_classes):
    img = Input(shape=(128, 128, 1))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(n_classes, 128 * 128)(label)
    label_embedding = Dense(128 * 128)(label_embedding)
    label_embedding = Reshape((128, 128, 1))(label_embedding)

    merged = Concatenate()([img, label_embedding])

    x = tfa.layers.SpectralNormalization(Conv2D(64, 4, strides=2, padding='same'))(merged)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = tfa.layers.SpectralNormalization(Conv2D(128, 4, strides=2, padding='same'))(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model([img, label], out)


# === Save samples ===
def save_generated_samples(generator, label_encoder, epoch, n_samples=5):
    os.makedirs(OUTPUT_GEN, exist_ok=True)
    n_classes = len(label_encoder.classes_)

    for i in range(n_samples):
        z = np.random.normal(0, 1, (1, LATENT_DIM))
        label_i = np.random.randint(0, n_classes, 1)
        gen_img = generator.predict([z, label_i], verbose=0)[0]

        real_img = ((gen_img + 1) * 127.5).astype(np.uint8).squeeze()
        class_name = label_encoder.inverse_transform(label_i)[0]

        output_folder = os.path.join(OUTPUT_GEN, class_name)
        os.makedirs(output_folder, exist_ok=True)

        img_path = os.path.join(output_folder, f"gen_epoch{epoch}_sample{i}.png")
        Image.fromarray(real_img).save(img_path)


# === Train the cGAN ===
def train_cgan(generator, discriminator, images, labels, epochs=1000, batch_size=32):
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    n_classes = len(le.classes_)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Compile discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0005))
    discriminator.trainable = False

    # Build combined model
    noise = Input(shape=(LATENT_DIM,))
    label = Input(shape=(1,), dtype='int32')
    generated_img = generator([noise, label])
    validity = discriminator([generated_img, label])
    combined = Model([noise, label], validity)
    combined.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0005))

    for epoch in range(1, epochs + 1):
        idx = np.random.randint(0, images.shape[0], batch_size)
        imgs = images[idx]
        lbls = labels_enc[idx].reshape(-1, 1)

        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        gen_imgs = generator.predict([noise, lbls], verbose=0)

        d_loss_real = discriminator.train_on_batch([imgs, lbls], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, lbls], fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        sampled_labels = np.random.randint(0, n_classes, batch_size).reshape(-1, 1)
        g_loss = combined.train_on_batch([noise, sampled_labels], real)

        print(f"Epoch {epoch}/{epochs} - D loss: {d_loss:.4f}, G loss: {g_loss:.4f}")

        if epoch % 100 == 0 or epoch == 1:
            save_generated_samples(generator, le, epoch)

    generator.save(MODEL_SAVE_PATH)
    print("✅ Training complete. Generator model saved to:", MODEL_SAVE_PATH)


# === Entry Point ===
if __name__ == "__main__":
    print("📦 Loading dataset...")
    X, y = load_images_labels(DATASET_PATH)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    n_classes = len(np.unique(y_train))
    print(f"✅ Loaded {X_train.shape[0]} images across {n_classes} classes.")

    generator = build_generator(LATENT_DIM, n_classes)
    discriminator = build_discriminator(n_classes)

    print("🚀 Starting training (Python 3.11.4 / TF 2.15)...")
    train_cgan(generator, discriminator, X_train, y_train, epochs=1000)

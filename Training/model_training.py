from __future__ import annotations
import cv2
from tensorflow.keras.utils import Sequence
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
from tqdm import tqdm
from typing import Callable, Sequence
from collections import defaultdict
import tensorflow as tf
import tensorflow_similarity as tfsim
from tensorflow.keras import layers
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.samplers import MultiShotMemorySampler, TFRecordDatasetSampler
from tensorflow_similarity.visualization import viz_neigbors_imgs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import os
import random
import itertools
import shutil
import math
import glob
from typing import Tuple
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import matplotlib.gridspec as gridspec
import pandas as pd
from pandas_ods_reader import read_ods
import csv
from sklearn.decomposition import PCA


batch_size = 64
patience = 10 #for early stopping

with open("evaluate.txt", 'w') as file:
               file.write("False")

### Begin: Image augmentation

def threshold(image):
    threshold_value = np.random.randint(
        160 - 20, 160 + 20)  # Slightly vary threshold value
    _, binary_image = cv2.threshold(
        image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def augaug(batch_img,dataset):
    if dataset == "real":
        augmented_batch = seq(images=batch_img)
    if dataset == "synth":
        augmented_batch = seq_synth(images=batch_img)
    for i in range(len(augmented_batch)):
        if np.random.uniform() < 0.5:
            augmented_batch[i] = threshold(augmented_batch[i].astype(np.uint8))
            augmented_batch[i] = cv2.GaussianBlur(
                augmented_batch[i].astype(np.uint8), (3, 3), 0.8)
    return augmented_batch

def get_subfolders(directory):
    return [os.path.join(directory, subfolder) for subfolder in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, subfolder))]

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale=(0.9, 1.0),
        translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
        rotate=(-25, 25),
        mode="constant",
        cval=255
    ),
], random_order=True)

seq_synth = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale=(0.9, 1.0),
        translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
        rotate=(-25, 25),
        mode="constant",
        cval=255
    ),
], random_order=True)

def batch_wide_scaling(batch_img):
    # Select a random scale factor for the entire batch
    scale_width = np.random.uniform(0.6,1)
    # Define an augmenter that applies the same scale to all images
    aug = iaa.ScaleX(scale_width,
        mode="constant",
        cval=255  # White background
    )
    augmented_batch = aug(images=batch_img)
    return augmented_batch

# def batch_wide_shear(batch_img):
#     # Define an augmenter that applies the same scale to all images
#     scale = np.random.uniform(0.01,0.05)
#     aug = iaa.PerspectiveTransform(scale=scale,
#         mode="constant",
#         cval=255  # White background
#     )
#     augmented_batch = aug(images=batch_img)
#     return augmented_batch

### End: image augmentation


def create_image_datasets(sets):
    datasets = []
    for folder in sets:
        datasets.append(tf.keras.utils.image_dataset_from_directory(
            folder,
            labels='inferred',
            shuffle=False,
            image_size=(32, 32),
            color_mode="grayscale"
        ))
    return datasets

#Real data (dinosaurs per occurrence and birds per species)
sets_real = get_subfolders("occurrences_v4")
#Synthetic data
sets_synth = get_subfolders("synthv4")

train_real = create_image_datasets(sets_real)
train_synth = create_image_datasets(sets_synth)

x_train_real = []
y_train_real = []

# Extract images and labels: Real dataset
for train_set in train_real:
    train_ds = train_set.unbatch()
    images = list(train_ds.map(
        tf.autograph.experimental.do_not_convert(lambda x, y: x)))
    labels = list(train_ds.map(
        tf.autograph.experimental.do_not_convert(lambda x, y: y)))
    train_real_images = np.array(images)[:, :, :, 0]  # Remove the grayscale channel
    train_real_labels = np.array(labels)
    x_train_real.append(train_real_images)
    y_train_real.append(train_real_labels)

x_train_synth = []
y_train_synth = []

# Extract images and labels: Synth dataset
for train_set in train_synth:
    train_ds = train_set.unbatch()
    images = list(train_ds.map(
        tf.autograph.experimental.do_not_convert(lambda x, y: x)))
    labels = list(train_ds.map(
        tf.autograph.experimental.do_not_convert(lambda x, y: y)))
    train_synth_images = np.array(images)[:, :, :, 0]  # Remove the grayscale channel
    train_synth_labels = np.array(labels)
    x_train_synth.append(train_synth_images)
    y_train_synth.append(train_synth_labels)

#Test dataset (validation dataset; designed to check for shortcut learning
test_ds = tf.keras.utils.image_dataset_from_directory(
    'validation/',
    labels='inferred',
    shuffle=False,
    image_size=(32, 32),
    color_mode="grayscale")

paths_te = test_ds.file_paths
paths_test = paths_te
for i in range(0, len(paths_test)):
    paths_test[i] = os.path.splitext(os.path.basename(paths_te[i]))[0]

test_ds = test_ds.unbatch()
test_images = list(test_ds.map(
    tf.autograph.experimental.do_not_convert(lambda x, y: x)))
test_labels = list(test_ds.map(
    tf.autograph.experimental.do_not_convert(lambda x, y: y)))
test_images = np.array(test_images)
test_labels = np.array(test_labels)

x_test = test_images[:, :, :, 0]
y_test = test_labels


#Sampler for validation data (collect batches of images)
def batch_sampler_validation(x, y, batch_size):
    num_samples = len(x)
    while True:
        indices = random.sample(range(num_samples), batch_size)
        batch_x = x[indices]
        batch_x = augaug(batch_x,dataset="real")
        batch_y = y[indices]
        yield tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)

#Sampler for training data (collect batches of images making sure at least 3 images per class are included)
def batch_sampler_train(x_train, y_train, batch_size,dataset):
    set_cycle = itertools.cycle(range(len(x_train)))  # Cycle through the sets (the subfolders of the training folder)
    while True:
        current_set_idx = next(set_cycle)  # Get the current set index
        #print(current_set_idx)
        x = x_train[current_set_idx]
        y = y_train[current_set_idx]
        # Step 1: Combine all images and labels into single lists for random selection
        all_images = list(x)
        all_labels = list(y)
        batch_x = []
        batch_y = []
        # Step 2: Fill the batch in chunks of 3 (since batch_size is a multiple of 3)
        while len(batch_x) < batch_size:
            # Randomly select an image (image index) from the entire dataset
            rand_index = random.randint(0, len(all_images) - 1)
            selected_image = all_images[rand_index]
            selected_class = all_labels[rand_index]
            same_class_images = [img for img, label in zip(all_images, all_labels) if label == selected_class]
            # Randomly select two more images from the same class
            if len(same_class_images) >= 3:
                selected_images = random.sample(same_class_images, 3)
            else:
                # If there are fewer than 3 images in the class, get more images
                selected_images = random.choices(same_class_images, k=3)
            # Add selected images and corresponding labels to batch
            batch_x.extend(selected_images)
            batch_y.extend([selected_class] * 3)
        # Step 3: Shuffle
        combined = list(zip(batch_x, batch_y))
        random.shuffle(combined)
        batch_x, batch_y = zip(*combined)
        # Step 4: augment the images
        #if dataset == "synth":
        #    if random.random() < 0.9:
        #        batch_x = batch_wide_scaling(batch_x)
        batch_x = augaug(np.array(batch_x),dataset)
        yield tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_y)

sampler_train_real = batch_sampler_train(x_train_real, y_train_real, batch_size,dataset="real")
sampler_train_synth = batch_sampler_train(x_train_synth, y_train_synth, batch_size,dataset="synth")
sampler_validation = batch_sampler_validation(x_test, y_test, batch_size)

# Model
inputs = tf.keras.layers.Input(shape=(32, 32, 1))
x = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)(inputs)
x = tf.keras.layers.Conv2D(64, 3, activation="elu")(x)
x = tf.keras.layers.Conv2D(64, 3, activation="elu")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation="elu")(x)
x = tf.keras.layers.Conv2D(128, 3, activation="elu")(x)
x = tf.keras.layers.Flatten()(x)
outputs = tfsim.layers.MetricEmbedding(128)(x)

model = tfsim.models.SimilarityModel(inputs, outputs)

#Initialize the model with weights from a previously trained model (tested but not used)
#model.load_weights('model_synthonly_20241101_134503.h5')

ms_loss_real = tfsim.losses.MultiSimilarityLoss(distance='cosine')
ms_loss_synth = tfsim.losses.MultiSimilarityLoss(distance='cosine')
#defaults: alpha 2.0, beta 40, lmda=0.5
#ms_loss_synth = tfsim.losses.TripletLoss(margin=0.1)   #soft margin

#Calculate steps per epoch
total_length = sum(len(element) for element in x_train_synth)
num_train_samples = int(total_length / len(x_train_synth))

num_validation_samples = len(x_test)
steps_per_epoch = num_train_samples // batch_size
validation_steps = num_validation_samples // batch_size
if validation_steps < 2:
    validation_steps = 1

#Plot 20 images (optinal, for checking only)
def pl(cmap='gray'):
    X_synth, y_synth = next(sampler_train_synth)
    # Create a subplot grid of 2 rows and 5 columns (for 10 images total)
    fig, axes = plt.subplots(5, 12, figsize=(10, 8))
    # Iterate over the first 10 images and display them
    for i, ax in enumerate(axes.flatten()):
        image = X_synth[i]
        # convert to numpy if necessary
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        ax.imshow(image, cmap=cmap)
        ax.axis('off')  # Turn off axis labels
    plt.tight_layout()
    plt.show()

# # #plot
# X_synth, y_synth = next(sampler_train_synth)
# tensor = X_synth[5]
# tensor_np = tensor.numpy()
# plt.imshow(tensor_np, cmap='gray')
# plt.show()
# #
# X_real, y_real = next(sampler_train_real)
# tensor = X_real[0]
# tensor_np = tensor.numpy()
# plt.imshow(tensor_np, cmap='gray')
# plt.show()


#Training loop that combines real and synthetic data
def custom_training_loop(model, sampler_train_real, sampler_train_synth,
                         steps_per_epoch, validation_data, validation_steps,
                         alpha, epochs, patience=patience, save_filepath='model.h5', optimizer=None):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    best_val_loss = float('inf')  # To track the best validation loss
    wait = 0  # Counter for patience
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        total_loss = 0
        # Training loop
        for step in range(steps_per_epoch):
            #Get a batch from the real and synthetic data samplers
            X_real, y_real = next(sampler_train_real)
            X_synth, y_synth = next(sampler_train_synth)
            with tf.GradientTape() as tape:
                y_pred_real = model(X_real, training=True)
                y_pred_synth = model(X_synth, training=True)
                #Compute losses
                loss_real = ms_loss_real(y_real, y_pred_real)  # Real data loss
                loss_synth = ms_loss_synth(y_synth, y_pred_synth)  # Synthetic data loss
                total_loss_batch = loss_real + (loss_synth * 2) # Combine losses; add weight
            #Compute gradients and apply them
            grads = tape.gradient(total_loss_batch, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            total_loss += total_loss_batch
        # Validate the model and Early Stopping
        validation_loss = 0
        for val_step in range(validation_steps):
            X_val, y_val = next(validation_data)
            y_pred_val = model(X_val, training=False)
            loss_val = ms_loss_real(y_val, y_pred_val)  # real data loss
            validation_loss += loss_val
        avg_val_loss = validation_loss / validation_steps
        print(f"Validation Loss for Epoch {epoch + 1}: {avg_val_loss.numpy()}")
        # Check if validation loss improved
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}.")
            best_val_loss = avg_val_loss
            wait = 0  # Reset patience counter
            model.save_weights(save_filepath)  # Save the best model
            with open("epoch.txt", 'w') as file:
               file.write(str(epoch))
        else:
            wait += 1
            print(f"No improvement in validation loss. Patience counter: {wait}/{patience}")
        # Early stopping: if no improvement, stop training
        if wait >= patience:
           print(f"Early stopping triggered. No improvement for {patience} epochs.")
           #write info
           with open("evaluate.txt", 'w') as file:
               file.write("True")
           break

custom_training_loop(model, sampler_train_real, sampler_train_synth,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=sampler_validation,
                     validation_steps=validation_steps,
                     alpha=0.7, epochs=200, patience=patience, save_filepath='model.h5')

with open("evaluate.txt", 'r') as file:
    evaluate = file.read()


if bool(evaluate) == True:

    with open("epoch.txt", 'r') as file:
        epoch = file.read()
        epoch = int(epoch)

    #create copy of model file with current date and time (to archive)
    save_filepath='model.h5'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name, extension = os.path.splitext(save_filepath)
    new_filepath = f"{base_name}_{timestamp}{extension}"
    shutil.copy(save_filepath, new_filepath)

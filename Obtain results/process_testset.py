import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob
import csv
from datetime import datetime
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Input, concatenate, Dropout
from tensorflow.keras.layers import Layer, BatchNormalization, MaxPooling2D, Concatenate, Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform, he_uniform
from tensorflow.keras.regularizers import l2
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
import math
from tensorflow.python.client import device_lib
import matplotlib.gridspec as gridspec
import pandas as pd
from pandas_ods_reader import read_ods
from tensorflow_similarity.models import SimilarityModel
from tensorflow.keras.models import load_model
import tensorflow_similarity as tfsim

#####
testset = "testset/"
#####

def count_folders(folder_path):
    try:
        entries = os.listdir(folder_path)
    except FileNotFoundError:
        print(f'Error: Directory "{folder_path}" not found.')
        return 0
    folder_count = 0
    for entry in entries:
        if os.path.isdir(os.path.join(folder_path, entry)):
            folder_count += 1
    return folder_count

test_ds = tf.keras.utils.image_dataset_from_directory(
    testset,
    labels="inferred",
    shuffle=False,
    image_size=(32, 32),
    color_mode="grayscale",
)

testimages = test_ds

paths_test_b = test_ds.file_paths
paths_test = paths_test_b
for i in range(0, len(paths_test)):
    paths_test[i] = os.path.splitext(os.path.basename(paths_test_b[i]))[0]

# separate images and labels, convert to numpy array
test_ds = test_ds.unbatch()
test_images = list(
    test_ds.map(tf.autograph.experimental.do_not_convert(lambda x, y: x))
)
test_labels = list(
    test_ds.map(tf.autograph.experimental.do_not_convert(lambda x, y: y))
)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

x_test = test_images
y_test = test_labels

#model
inputs = tf.keras.layers.Input(shape=(32, 32, 1))
x = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)(inputs)
x = tf.keras.layers.Conv2D(64, 3, activation="elu")(x)
x = tf.keras.layers.Conv2D(64, 3, activation="elu")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation="elu")(x)
x = tf.keras.layers.Conv2D(128, 3, activation="elu")(x)
x = tf.keras.layers.Flatten()(x)
outputs = tfsim.layers.MetricEmbedding(128)(x)

embedding_model = SimilarityModel(inputs, outputs)

embedding_model.load_weights('model_approach_4.h5')

import umap
import matplotlib.pyplot as plt
# set mode to 'markers+text' to have point labels
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px

#testset
embeddings = embedding_model.predict(testimages)
testsetdot = testset + "."
test_class_labels = np.unique(np.array(y_test))
reallabels = next(os.walk(testsetdot))[1]
reallabels = sorted(reallabels)
class_labels = y_test

label_map = {i: reallabels[i] for i in range(len(reallabels))}
labelss = np.array([label_map[label] for label in class_labels])

n_classes = count_folders(testset)
n_colors = n_classes
color_scale = px.colors.sample_colorscale(
    "rainbow", [n / (n_colors - 1) for n in range(n_colors)]
)

# UMAP visualisation
umap_model = umap.UMAP(n_components=2, min_dist=0.4, n_neighbors=10)
umap_embeddings_transformed = umap_model.fit_transform(embeddings)

data = []
point_labels = paths_test + [f"Mean_{label}" for label in labelss]
unique_labels = np.unique(labelss)

testset_embeddings_transformed = umap_embeddings_transformed[:len(embeddings)]

for i, class_label in enumerate(reallabels):
    testset_mask = labelss == class_label
    testset_indices = np.where(testset_mask)[0]
    testset_trace = go.Scatter(
        x=testset_embeddings_transformed[testset_indices, 0],
        y=testset_embeddings_transformed[testset_indices, 1],
        mode="markers",
        marker=dict(color=color_scale[i], size=9, symbol="circle"),
        text=[point_labels[j] for j in testset_indices],
        textposition="bottom center",
        name=str(class_label),
    )
    data.append(testset_trace)

# Layout and Plot
layout = go.Layout(title="UMAP Embedding", hovermode="closest", showlegend=True)
fig = go.Figure(data=data, layout=layout)
pio.write_html(fig, file="umap_plot.html", auto_open=True)
fig.write_image("umap_plot.svg", width=1200, height=600)

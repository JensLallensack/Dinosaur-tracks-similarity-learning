import os
import numpy as np
import tensorflow as tf
import tensorflow_similarity as tfsim
from tensorflow.keras import layers
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow.keras.utils import img_to_array, load_img
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
IMG_SIZE = (32, 32)
NUM_SPLITS = 10
DATA_DIR = './'
number_classes = None
model_path = 'model_main_approach.h5'

# Load images
def load_images_from_directory(base_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(base_dir))
    for class_name in class_names:
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.endswith('.png'):
                img_path = os.path.join(class_path, filename)
                img = load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0  # normalise
                images.append(img_array)
                labels.append(class_name)
    return np.array(images), np.array(labels)

f1_scores_macro = []
cumulative_conf_mat = None

for i in range(1, NUM_SPLITS + 1):
    print(f"\nProcessing split {i}...")
    split_dir = os.path.join(DATA_DIR, f"split{i}")
    train_dir = os.path.join(split_dir, 'train')
    test_dir = os.path.join(split_dir, 'test')
    # Load model
    inputs = tf.keras.layers.Input(shape=(32, 32, 1))
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="elu")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="elu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="elu")(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="elu")(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tfsim.layers.MetricEmbedding(128)(x)
    embedding_model = tfsim.models.SimilarityModel(inputs, outputs)
    embedding_model.load_weights(model_path)
    # Load images
    x_train_raw, y_train = load_images_from_directory(train_dir)
    x_test_raw, y_test = load_images_from_directory(test_dir)
    x_train_raw = x_train_raw.reshape((-1, 32, 32, 1))
    x_test_raw = x_test_raw.reshape((-1, 32, 32, 1))
    # Load labels
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    class_names = le.classes_
    if number_classes is None:
        number_classes = len(class_names)
        cumulative_conf_mat = np.zeros((number_classes, number_classes), dtype=np.float32)
    # Generate embeddings
    x_train_embed = embedding_model.predict(x_train_raw, batch_size=64)
    x_test_embed = embedding_model.predict(x_test_raw, batch_size=64)
    # Scale embeddings
    scaler = StandardScaler()
    x_train_embed = scaler.fit_transform(x_train_embed)
    x_test_embed = scaler.transform(x_test_embed)
    # Train classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=32)
    classifier.fit(x_train_embed, y_train_enc)
    # Classify mean embeddings
    y_true_mean = []
    y_pred_mean = []
    for class_index in range(number_classes):
        # Find all test embeddings belonging to current class
        class_mask = (y_test_enc == class_index)
        class_embeddings = x_test_embed[class_mask]
        if len(class_embeddings) > 0:
            # Compute mean embedding for class and predict
            mean_embedding = np.mean(class_embeddings, axis=0).reshape(1, -1)
            predicted_class_enc = classifier.predict(mean_embedding)[0]
            y_true_mean.append(class_index)
            y_pred_mean.append(predicted_class_enc)
    if y_true_mean:
        conf_mat_split = confusion_matrix(y_true_mean, y_pred_mean, labels=range(number_classes))
        f1_macro_split = f1_score(y_true_mean, y_pred_mean, average='macro', zero_division=0)
        cumulative_conf_mat += conf_mat_split
        f1_scores_macro.append(f1_macro_split)
        print(f"Split {i} - Macro F1: {f1_macro_split:.4f}")

# Plot
avg_conf_mat = cumulative_conf_mat / NUM_SPLITS
avg_f1_macro = np.mean(f1_scores_macro)
print("\nAverage Per-class Accuracy:")
for i, class_name in enumerate(class_names):
    # Per-class accuracy
    if avg_conf_mat[i].sum() > 0:
        class_accuracy = avg_conf_mat[i, i] / avg_conf_mat[i].sum()
    else:
        class_accuracy = 0
    print(f"{class_name}: {class_accuracy:.2%}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(avg_conf_mat, annot=True, fmt='.1f', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Average Confusion Matrix (Mean Embeddings) Over 10 Splits")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion.svg", format="svg")
plt.show()

# Macro F1 score
print(f"\nAverage Macro F1 Score over {NUM_SPLITS} splits: {avg_f1_macro:.4f}")

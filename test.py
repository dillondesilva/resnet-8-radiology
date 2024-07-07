from resnet8 import *
from tqdm import tqdm

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

argparser = argparse.ArgumentParser(
    prog='ResNet18 Test Module',
    description='Script for testing ResNet18 on brain tumor image dataset',
)

argparser.add_argument('weights')

args = argparser.parse_args()
path_to_weights = args.weights

builder = tfds.ImageFolder('./BrainScanData/')
print(builder.info)  # num examples, labels... are automatically calculated

my_resnet = ResNet8(n_classes=4)
final_model_checkpoint = tf.train.Checkpoint(model=my_resnet)
latest_checkpoint = tf.train.latest_checkpoint(path_to_weights)
final_model_checkpoint.restore(latest_checkpoint)


test_ds = builder.as_dataset(split='test')
true_positives = 0
n_predictions = 0
labels = []
predictions = []

for item in tqdm(test_ds):
    image = tf.cast(item["image"], dtype=tf.float32)
    image = tf.reshape(image, shape=((1,) + image.shape))
    label = item["label"]
    prediction = np.argmax(tf.nn.softmax(my_resnet(image)))
    if prediction == label:
        true_positives += 1
    
    labels.append(label)
    predictions.append(prediction)
    n_predictions += 1

print(f"Validation accuracy is {true_positives / n_predictions}")

conf_matrix = tf.math.confusion_matrix(
    labels,
    predictions,
    num_classes=4
)

conf_matrix_np = conf_matrix.numpy()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_np, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()

test_ds = builder.as_dataset(split='train')
true_positives = 0
n_predictions = 0
labels = []
predictions = []

for item in tqdm(test_ds):
    image = tf.cast(item["image"], dtype=tf.float32)
    image = tf.reshape(image, shape=((1,) + image.shape))
    label = item["label"]
    prediction = np.argmax(tf.nn.softmax(my_resnet(image)))
    if prediction == label:
        true_positives += 1
    
    labels.append(label)
    predictions.append(prediction)
    n_predictions += 1

print(f"Training accuracy is {true_positives / n_predictions}")

conf_matrix = tf.math.confusion_matrix(
    labels,
    predictions,
    num_classes=4
)

conf_matrix_np = conf_matrix.numpy()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_np, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()
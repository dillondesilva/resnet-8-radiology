from resnet8 import *
from tqdm import tqdm

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os


argparser = argparse.ArgumentParser(
    prog='ResNet18 Trainer',
    description='Script for training a ResNet18 ',
)

argparser.add_argument('name')
argparser.add_argument('epochs')
argparser.add_argument('--last-checkpoint')

# Parsing model name arguments and number of epochs
args = argparser.parse_args()
epochs = int(args.epochs)
model_name = args.name

# Create directory to store model weights in
if not os.path.exists(model_name):
    os.makedirs(model_name)
    print(f"Directory '{model_name}' created.")
else:
    print(f"Directory '{model_name}' already exists.")

# Load our data for training
builder = tfds.ImageFolder('./BrainScanData/')
print(builder.info)
ds = builder.as_dataset(split='train', shuffle_files=True)
tfds.show_examples(ds, builder.info)
n_classes = 4
my_resnet = ResNet8(n_classes=n_classes)
print(f"Number of layers: {len(my_resnet.trainable_variables)}")
input = tf.random.uniform((1, 64, 64, 3))
target = tf.Variable([0, 0, 1, 0], dtype=tf.float32)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
print(optimizer.learning_rate.numpy())

if args.last_checkpoint:
    print(f"LOADING CHECKPOINT: {args.last_checkpoint}")
    final_model_checkpoint = tf.train.Checkpoint(model=my_resnet)
    latest_checkpoint = tf.train.latest_checkpoint(args.last_checkpoint)
    final_model_checkpoint.restore(latest_checkpoint)
    print("CHECKPOINT LOADED")

def validate_accuracy(accuracies):
    test_ds = builder.as_dataset(split='test', shuffle_files=True)
    true_positives = 0
    n_predictions = 0
    for item in test_ds:
        image = tf.cast(item["image"], dtype=tf.float32)
        image = tf.reshape(image, shape=((1,) + image.shape))
        label = item["label"]
        prediction = np.argmax(tf.nn.softmax(my_resnet(image)))
        if prediction == label:
            true_positives += 1
        
        n_predictions += 1

    accuracies.append(true_positives / n_predictions)
    print(f"Accuracy is {true_positives / n_predictions}")

# Training loop
tf.device("GPU")

losses = []
accuracies = []
n_items = len(ds)
# with tf.device("GPU"):
for epoch in tqdm(range(epochs)):
    epoch_loss_count = 0
    print(f"--- STARTING EPOCH {epoch} ---")
    for idx, item in tqdm(enumerate(ds)):
        image = tf.cast(item["image"], dtype=tf.float32)
        image = tf.reshape(image, shape=((1,) + image.shape))
        label = tf.reshape(tf.one_hot(item["label"], n_classes), shape=(1,n_classes))
        label = tf.cast(label, dtype=tf.float32)
        loss = train_step(my_resnet, image, label, optimizer)
        epoch_loss_count += loss.numpy()

    losses.append(epoch_loss_count / len(ds))
    # Ideally this should be a percentage but its roughly 33% of the test set
    validate_accuracy(accuracies)

    # Create a checkpoint
    checkpoint = tf.train.Checkpoint(model=my_resnet)
    checkpoint.save(f"./{model_name}/resnet18-{model_name}-epoch-{str(epoch)}")

# Write to .csv, plots, etc
epoch_labels = np.arange(1, epochs + 1, 1)
training_metrics = {
    "Epochs": epoch_labels,
    "Accuracy": accuracies,
    "Loss": losses
}

training_metrics_df = pd.DataFrame(training_metrics)
training_metrics_df.to_csv(f"{model_name}/{model_name}-training-metrics.csv")

plt.style.use('ggplot')
plt.title("Training Loss")
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epoch")
plt.plot(epoch_labels, losses)
plt.show()

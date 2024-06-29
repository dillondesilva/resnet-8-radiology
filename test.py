from resnet18 import *
from tqdm import tqdm

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


builder = tfds.ImageFolder('./Data/')
print(builder.info)  # num examples, labels... are automatically calculated

my_resnet = ResNet18(n_classes=4)
final_model_checkpoint = tf.train.Checkpoint(model=my_resnet)
final_model_checkpoint.restore("brain_scan_resnet18model-1.index")


test_ds = builder.as_dataset(split='test', shuffle_files=True)
true_positives = 0
n_predictions = 0

for item in tqdm(test_ds):
    image = tf.cast(item["image"], dtype=tf.float32)
    image = tf.reshape(image, shape=((1,) + image.shape))
    label = item["label"]
    prediction = np.argmax(tf.nn.softmax(my_resnet(image)))
    if prediction == label:
        true_positives += 1
    
    n_predictions += 1

print(f"Accuracy is {true_positives / n_predictions}")

# ResNet-8 in Tensorflow Core

This was a small hobby project I did to have a crack at implementing a network architecture more bare-bones (opposed to using existing, out of the box models). Here's a guide to the key training/application scripts available.

**`resnet-8.py`**: This contains a class for an 8-layer ResNet designed to classify images with a 256 x 256 resolution. In this file, you can find my implementation of the residual block and train step.

**`train.py`**: This script contains a simple, iterative training loop over the image data. It will dump checkpoints and metrics from training into a folder that can be used for later debugging/post-processing.

**`test.py`**: Loads the specified weights into the ResNet-8 model and outputs test accuracy.

**`dataset_helper.py`**: This is a utility script which takes the [original dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and converts it into the expected structure for training/test scripts to work

**`resize_images.py`**: Traverses a directory and resizes all images, deleting any corrupt items in the process

**`app.py`**: Streamlit app that provides an interface to the model and run predictions on images.

![training_metrics_plot](https://github.com/user-attachments/assets/fca92fbc-73d3-41ad-a8d5-3cdf5da4d8de)

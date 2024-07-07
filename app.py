from resnet8 import *
from PIL import Image

import streamlit as st
import tensorflow as tf
import numpy as np

image_to_classify = None
image_holder = st.empty()
image_file = None

def load_model(model_instance, path_to_weights):
    # Technically loading a training checkpoint
    # but this can change in future
    final_model_checkpoint = tf.train.Checkpoint(model=model_instance)
    latest_checkpoint = tf.train.latest_checkpoint(path_to_weights)
    final_model_checkpoint.restore(latest_checkpoint)

def pred_to_label(pred):
    match pred:
        case 0:
            return "Glioma"
        case 1:
            return "Meningioma"
        case 2:
            return "No tumor detected"
        case 3:
            return "Pituitary"
        case _:
            return "Error occurred when classifying tumor"

def classify_image(model_instance):
    global image_file
    global image_holder

    # Resize image
    with Image.open(image_file) as img:
        # Resize the image
        print(img)
        img = img.convert('RGB')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_array  = tf.keras.utils.img_to_array(img)
        img_array = np.reshape(img_array, (1,) + img_array.shape)
        print(f"IMGARRAY: {img_array.shape}")

        prediction = np.argmax(tf.nn.softmax(model_instance(img_array)))
        label = pred_to_label(prediction)
        st.markdown(f"Detected tumor is: {label}")
        st.image(image_file, width=80)

def main():
    global image_to_classify
    global image_holder
    global image_file
    my_resnet = ResNet8(n_classes=4)
    load_model(my_resnet, "resnet-8-brain-tumor-classification-5-epoch/final-checkpoint")
    # Create GUI elements
    st.title("ResNet-8 Implementation for Brain Tumor Classification")
    st.markdown("Powered by a ResNet-8 deep learning model, this app allows you to upload a brain scan and classifies up to 4 classes of tumor including Meningioma, Pituitary, Glioma or Non-existence of tumor.")
    st.markdown("Training data was sourced from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)")
    image_file = st.file_uploader("Upload a brain tumor image (Normal, Meningioma, Pituitary, Glioma)")
    st.button("Classify Image", on_click=classify_image, args=(my_resnet,))

main()
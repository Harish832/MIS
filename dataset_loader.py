import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset path
dataset_path = "./Medicinal Leaf dataset"

def load_dataset():
    """Function to load dataset and display class details in Streamlit"""
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path '{dataset_path}' not found!")
        return

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        shuffle=True,
        batch_size=32,
        image_size=(299, 299),
    )

    labels = dataset.class_names
    st.write("### Class Labels:", labels)

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_dataset = datagen.flow_from_directory(
        dataset_path, target_size=(299, 299), batch_size=32, class_mode='sparse', subset='training'
    )

    val_dataset = datagen.flow_from_directory(
        dataset_path, target_size=(299, 299), batch_size=32, class_mode='sparse', subset='validation'
    )

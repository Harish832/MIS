import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# Define dataset path
dataset_path = "./Medicinal Leaf dataset"

def efficient_net_b0():
    """Load dataset and display class details in Streamlit"""
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path '{dataset_path}' not found!")
        return

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,  # Reduce augmentation to speed up
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    train_dataset = datagen.flow_from_directory(
        dataset_path, target_size=(224, 224), batch_size=16, class_mode='sparse', subset='training'
    )

    val_dataset = datagen.flow_from_directory(
        dataset_path, target_size=(224, 224), batch_size=16, class_mode='sparse', subset='validation'
    )

    st.write("### 🚀 Training Model with EfficientNetB0")

    # EfficientNetB0 as Base Model
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    base_model.trainable = False  # Freeze base model layers

    model = models.Sequential([
        base_model,
        layers.Dense(256, activation='relu'),  # Reduced dense layers
        layers.Dropout(0.3),
        layers.Dense(len(train_dataset.class_indices), activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early Stopping Callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Lower patience

    # Train the model for fewer epochs
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2,  # Reduced from 25 to 5
        steps_per_epoch=5,  # Limit training steps
        validation_steps=2,  # Reduce validation steps
        callbacks=[early_stopping],
        verbose=1
    )

    # Display final accuracy
    final_training_accuracy = history.history['accuracy'][-1]
    final_validation_accuracy = history.history['val_accuracy'][-1]

    st.write(f"✅ **Final Training Accuracy:** {final_training_accuracy:.4f}")
    st.write(f"✅ **Final Validation Accuracy:** {final_validation_accuracy:.4f}")

    # Plot Training History
    st.write("### 📈 Training History")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    
    st.pyplot(fig)

efficient_net_b0()

import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.title("🚀 EfficientNetB3 Model Training")

# Upload dataset
dataset_path = "./Medicinal Leaf dataset"

st.success("✅ Dataset found!")

@st.cache_resource
def load_data(dataset_path):
    """Preprocess dataset with caching to reduce reload time."""
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        dataset_path, target_size=(224, 224), batch_size=64, 
        subset='training', class_mode='categorical'  # **Changed from sparse to categorical**
    )

    val_data = datagen.flow_from_directory(
        dataset_path, target_size=(224, 224), batch_size=64, 
        subset='validation', class_mode='categorical'  # **Changed here too**
    )

    return train_data, val_data

def build_model(num_classes):
    """Constructs an optimized EfficientNetB3 model."""
    base_model = EfficientNetB3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base layers

    model = models.Sequential([
        base_model,
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # **Changed loss function**
        metrics=['accuracy']
    )
    return model

def efficient_net_b3() :
    train_data, val_data = load_data(dataset_path)
    num_classes = len(train_data.class_indices)
    model = build_model(num_classes)

    # Early Stopping Callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Training Button
    if st.button("🚀 Start Training"):
        st.write("⏳ **Training in progress... Please wait!**")
        progress_bar = st.progress(0)  # Training progress bar
        
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=2,  # Reduced from 25 to 5
            steps_per_epoch=2,  # Limit training steps
            validation_steps=2,  # Reduce validation steps
            callbacks=[early_stopping]
        )

        st.success("✅ Training Completed!")
        
        # Update progress bar to 100%
        progress_bar.progress(100)

        # Plot Accuracy Graph
        st.write("### 📊 Training Progress")
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

        # Save Model
        model.save("trained_model.keras")
        with open("trained_model.keras", "rb") as f:
            st.download_button("💾 Download Model", f, file_name="trained_model.keras")

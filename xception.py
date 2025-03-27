import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.title("🚀 Xception Model Training")

# 📂 Dataset Input
dataset_path = "./Medicinal Leaf dataset"

if not dataset_path or not os.path.exists(dataset_path):
    st.warning("⚠️ Please enter a valid dataset path.")
    st.stop()

st.success("✅ Dataset found!")

# 🏗️ Function to Load Data
@st.cache_resource
def load_data(dataset_path):
    """Loads and preprocesses dataset with caching to improve speed."""
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        dataset_path, target_size=(299, 299), batch_size=64, 
        subset='training', class_mode='categorical'  # ✅ Changed to categorical
    )

    val_data = datagen.flow_from_directory(
        dataset_path, target_size=(299, 299), batch_size=64, 
        subset='validation', class_mode='categorical'  # ✅ Changed to categorical
    )

    return train_data, val_data

# 🏗️ Build Optimized Xception Model
def build_xception_model(num_classes):
    """Constructs an optimized Xception model."""
    xception_base = Xception(
        input_shape=(299, 299, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    xception_base.trainable = False  # Freeze base layers

    model = models.Sequential([
        xception_base,
        layers.BatchNormalization(),  # ✅ Better convergence
        layers.Dense(512, activation='relu'),  # ✅ Increased neurons
        layers.Dropout(0.4),  # ✅ Increased dropout
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # ✅ Changed from sparse to categorical
        metrics=['accuracy']
    )
    return model

def xception():
    train_data, val_data = load_data(dataset_path)
    num_classes = len(train_data.class_indices)
    model = build_xception_model(num_classes)

    # ⏳ Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # 🚀 Training Button
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

        # 📊 Plot Accuracy Graph
        st.write("### 📊 Training Progress")
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

        # 💾 Save Model
        model.save("xception_trained_model.h5")  # ✅ Using .h5 format (smaller file size)
        with open("xception_trained_model.h5", "rb") as f:
            st.download_button("💾 Download Model", f, file_name="xception_trained_model.h5")

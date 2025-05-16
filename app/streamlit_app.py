import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
from neural_network import NeuralNetwork
import tensorflow as tf

def deploy_app():    
    st.set_page_config(page_title="MNIST Classifier", layout="wide")
    st.title("MNIST Handwritten Digit Classifier")
    
    # Check for available models
    available_models = []
    model_files = ["saved_models/bp_model.pkl", "saved_models/jade_model.pkl", "saved_models/de_model.pkl", "saved_models/ga_model.pkl", "saved_models/pso_model.pkl"]
    model_names = ["Backpropagation", "JADE", "Differential Evolution", "Genetic Algorithm", "Particle Swarm Optimization"]
    
    for file, name in zip(model_files, model_names):
        if os.path.exists(file):
            available_models.append((file, name))
    
    if not available_models:
        st.error("No trained models found! Please run the training scripts first.")
        return
    
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    selected_model_file, selected_model_name = st.sidebar.selectbox(
        "Choose a model to use:",
        available_models,
        format_func=lambda x: x[1]
    )
    
    st.sidebar.write(f"Using {selected_model_name} model")
    
    # Load the selected model
    model = NeuralNetwork.load_model(selected_model_file)
    
    st.write("Choose Random Digit From the Dataset")
    
    # Sample MNIST digit
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    sample_index = st.slider("Sample digit index (for demonstration):", 0, 300, 0)
    sample_digit = x_test[sample_index].reshape(28, 28)
    
    st.image(sample_digit, caption=f"Sample Digit (Label: {y_test[sample_index]})", width=150)
    
    # Normalize and reshape for prediction
    sample_input = sample_digit.reshape(1, 784).astype(np.float32) / 255.0
    
    # Make prediction
    digit = model.predict(sample_input)[0]
    
    st.success(f"Prediction: {digit}")
    
    # Display confidence scores
    probs, _, _ = model.forward(sample_input)
    
    # Plot confidence scores
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(10), probs[0])
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
    
    # Model information
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"Hidden Units: {model.hidden_dim}")
    st.sidebar.write(f"Input Dimension: {model.input_dim}")
    st.sidebar.write(f"Output Classes: {model.output_dim}")
    
    
deploy_app()
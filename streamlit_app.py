import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Autoencoders Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load Keras models for inference. Use compile=False to avoid metric deserialization issues.

    Falls back to attempting a load with simple custom_objects if needed.
    """
    model_paths = {
        'cnn_ae': 'models/cnn_autoencoder.h5',
        'cnn_enc': 'models/cnn_encoder.h5',
        'lstm_ae': 'models/lstm_autoencoder.h5',
        'lstm_enc': 'models/lstm_encoder.h5'
    }

    # Check files exist first
    missing = [p for p in model_paths.values() if not os.path.exists(p)]
    if missing:
        st.error(f"❌ Models not found: {', '.join(missing)}\nRun the notebook to train/save models or provide them via Git LFS.")
        return None, None, None, None

    try:
        # Primary load: compile=False avoids attempting to deserialize metrics/losses
        cnn_ae = keras.models.load_model(model_paths['cnn_ae'], compile=False)
        cnn_enc = keras.models.load_model(model_paths['cnn_enc'], compile=False)
        lstm_ae = keras.models.load_model(model_paths['lstm_ae'], compile=False)
        lstm_enc = keras.models.load_model(model_paths['lstm_enc'], compile=False)
        
        # Recompile for inference
        cnn_ae.compile(optimizer='adam', loss='mse', metrics=['mae'])
        lstm_ae.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return cnn_ae, cnn_enc, lstm_ae, lstm_enc
    except Exception as e:
        # Fallback: some older save formats reference functions by name. Try lightweight custom_objects.
        try:
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.losses.MeanAbsoluteError(),
            }
            cnn_ae = keras.models.load_model(model_paths['cnn_ae'], custom_objects=custom_objects, compile=False)
            cnn_enc = keras.models.load_model(model_paths['cnn_enc'], compile=False)
            lstm_ae = keras.models.load_model(model_paths['lstm_ae'], custom_objects=custom_objects, compile=False)
            lstm_enc = keras.models.load_model(model_paths['lstm_enc'], compile=False)
            return cnn_ae, cnn_enc, lstm_ae, lstm_enc
        except Exception as e2:
            st.error(f"❌ Error loading models: {e2}")
            return None, None, None, None

@st.cache_data
def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train[:10000], x_test[:2000], y_train[:10000], y_test[:2000]

@st.cache_data
def generate_timeseries_data(n_samples=500):
    """Generate synthetic time-series data"""
    np.random.seed(42)
    sequence_length = 50
    n_features = 10
    
    data = []
    for _ in range(n_samples):
        seq = np.sin(np.arange(sequence_length).reshape(-1, 1) * 
                     np.random.uniform(0.1, 0.5, n_features)) + \
              np.cos(np.arange(sequence_length).reshape(-1, 1) * 
                     np.random.uniform(0.1, 0.5, n_features))
        seq += np.random.normal(0, 0.1, (sequence_length, n_features))
        data.append(seq)
    
    data = np.array(data)
    data = (data - data.mean(axis=(0, 1))) / (data.std(axis=(0, 1)) + 1e-6)
    return data, sequence_length, n_features

# Title and Header
st.title("🧠 Autoencoders Lab: CNN vs LSTM")
st.markdown("---")
st.markdown("""
Explore and compare **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM)** 
autoencoders for feature extraction and dimensionality reduction.
""")

# Sidebar navigation
st.sidebar.title("🗂️ Navigation")
page = st.sidebar.radio("Select a section:", 
    ["🏠 Home", "🖼️ CNN Autoencoder", "📈 LSTM Autoencoder", "📊 Comparison", "📋 Analysis"])

# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    st.header("Welcome to the Autoencoders Lab")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 What are Autoencoders?")
        st.markdown("""
        Autoencoders are neural networks designed for unsupervised learning that learn to 
        compress data and then reconstruct it. They consist of two main parts:
        
        - **Encoder**: Compresses input data into a lower-dimensional latent representation
        - **Decoder**: Reconstructs the original data from the latent representation
        
        **Applications:**
        - Dimensionality reduction
        - Image denoising
        - Anomaly detection
        - Feature extraction
        - Data compression
        """)
    
    with col2:
        st.subheader("🔬 Lab Objectives")
        st.markdown("""
        1. **Understand** autoencoder concepts and applications
        2. **Implement** CNN autoencoders for spatial data (images)
        3. **Implement** LSTM autoencoders for temporal data (sequences)
        4. **Compare** performance metrics and characteristics
        5. **Analyze** latent space representations
        """)
    
    st.markdown("---")
    
    # Key metrics overview
    st.subheader("📊 Quick Statistics")
    
    cnn_ae, cnn_enc, lstm_ae, lstm_enc = load_models()
    
    if cnn_ae is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CNN Test Loss (MSE)", "0.015949")
        
        with col2:
            st.metric("CNN Test MAE", "0.092442")
        
        with col3:
            st.metric("CNN Latent Dim", "32")
        
        with col4:
            st.metric("CNN Compression", "96.0x")

# ============================================
# CNN AUTOENCODER PAGE
# ============================================
elif page == "🖼️ CNN Autoencoder":
    st.header("CNN Autoencoder for Image Data (CIFAR-10)")
    
    cnn_ae, cnn_enc, lstm_ae, lstm_enc = load_models()
    
    if cnn_ae is None:
        st.error("Models not loaded. Please train them first.")
    else:
        x_train, x_test, y_train, y_test = load_cifar10_data()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Architecture", "Reconstruction", "Latent Space", "Metrics"])
        
        # Tab 1: Architecture
        with tab1:
            st.subheader("Model Architecture")
            st.markdown("""
            **Encoder:**
            - Conv2D(32) + MaxPool → Conv2D(64) + MaxPool → Conv2D(128)
            - Global Average Pooling → Dense(32) [Latent Space]
            
            **Decoder:**
            - Dense → Reshape → Conv2DTranspose(128)
            - UpSampling2D → Conv2DTranspose(64)
            - UpSampling2D → Conv2DTranspose(3, sigmoid)
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Encoder Output Shape:** (None, 32)")
            with col2:
                st.info("**Decoder Output Shape:** (None, 32, 32, 3)")
        
        # Tab 2: Reconstruction
        with tab2:
            st.subheader("Image Reconstruction Comparison")
            
            n_images = st.slider("Number of images to display:", 3, 10, 5)
            
            reconstructed = cnn_ae.predict(x_test[:n_images], verbose=0)
            
            fig, axes = plt.subplots(2, n_images, figsize=(14, 3))
            
            for i in range(n_images):
                axes[0, i].imshow(x_test[i])
                axes[0, i].set_title(f'Original', fontsize=9)
                axes[0, i].axis('off')
                
                axes[1, i].imshow(reconstructed[i])
                axes[1, i].set_title(f'Reconstructed', fontsize=9)
                axes[1, i].axis('off')
            
            plt.suptitle('Original vs Reconstructed Images', fontsize=12)
            st.pyplot(fig)
            plt.close()
        
        # Tab 3: Latent Space
        with tab3:
            st.subheader("Latent Space Visualization")
            
            st.info("⏳ This may take a moment to compute...")
            
            latent_rep = cnn_enc.predict(x_test[:500], verbose=0)
            labels = y_test[:500].flatten()
            
            # PCA
            pca = PCA(n_components=2)
            latent_pca = pca.fit_transform(latent_rep)
            
            fig = px.scatter(
                x=latent_pca[:, 0],
                y=latent_pca[:, 1],
                color=labels,
                title="CNN Latent Space (PCA)",
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                        'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Metrics
        with tab4:
            st.subheader("Performance Metrics")
            
            # Use pre-computed metrics from training
            test_loss = 0.015949
            test_mae = 0.092442
            
            metrics_data = {
                'Metric': ['MSE Loss', 'MAE', 'RMSE', 'Compression Ratio', 'Compression %'],
                'Value': [
                    f'{test_loss:.6f}',
                    f'{test_mae:.6f}',
                    f'{np.sqrt(test_loss):.6f}',
                    '96.0x',
                    '98.96%'
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Bar(x=['CNN'], y=[test_loss])])
                fig.update_layout(title="Test MSE Loss", yaxis_title="Loss")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Bar(x=['CNN'], y=[test_mae])])
                fig.update_layout(title="Test MAE", yaxis_title="MAE")
                st.plotly_chart(fig, use_container_width=True)

# ============================================
# LSTM AUTOENCODER PAGE
# ============================================
elif page == "📈 LSTM Autoencoder":
    st.header("LSTM Autoencoder for Sequential Data")
    
    cnn_ae, cnn_enc, lstm_ae, lstm_enc = load_models()
    
    if lstm_ae is None:
        st.error("Models not loaded.")
    else:
        timeseries_data, seq_len, n_feat = generate_timeseries_data(500)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Architecture", "Reconstruction", "Latent Space", "Metrics"])
        
        # Tab 1: Architecture
        with tab1:
            st.subheader("Model Architecture")
            st.markdown("""
            **Encoder:**
            - LSTM(64, return_sequences=True)
            - LSTM(32, return_sequences=False)
            - Dense(20) [Latent Space]
            
            **Decoder:**
            - RepeatVector(sequence_length)
            - LSTM(32, return_sequences=True)
            - LSTM(64, return_sequences=True)
            - TimeDistributed(Dense(features))
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Input Shape:** ({seq_len}, {n_feat})")
            with col2:
                st.info("**Latent Dimension:** 20")
        
        # Tab 2: Reconstruction
        with tab2:
            st.subheader("Sequence Reconstruction")
            
            n_seq = st.slider("Number of sequences:", 1, 5, 3)
            reconstructed_seq = lstm_ae.predict(timeseries_data[:n_seq], verbose=0)
            
            fig, axes = plt.subplots(n_seq, 1, figsize=(12, 3*n_seq))
            
            if n_seq == 1:
                axes = [axes]
            
            for i in range(n_seq):
                axes[i].plot(timeseries_data[i, :, 0], 'b-', linewidth=2, alpha=0.7, label='Original')
                axes[i].plot(reconstructed_seq[i, :, 0], 'r--', linewidth=2, alpha=0.7, label='Reconstructed')
                axes[i].set_ylabel('Value')
                axes[i].set_title(f'Sequence {i+1} (Feature 0)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Timestep')
            st.pyplot(fig)
            plt.close()
        
        # Tab 3: Latent Space
        with tab3:
            st.subheader("LSTM Latent Space Visualization")
            
            latent_rep_lstm = lstm_enc.predict(timeseries_data[:500], verbose=0)
            
            # PCA
            pca_lstm = PCA(n_components=2)
            latent_pca_lstm = pca_lstm.fit_transform(latent_rep_lstm)
            
            labels_ts = np.arange(len(latent_pca_lstm)) % 10
            
            fig = px.scatter(
                x=latent_pca_lstm[:, 0],
                y=latent_pca_lstm[:, 1],
                color=labels_ts,
                title="LSTM Latent Space (PCA)",
                labels={'x': f'PC1 ({pca_lstm.explained_variance_ratio_[0]:.1%})',
                        'y': f'PC2 ({pca_lstm.explained_variance_ratio_[1]:.1%})'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Metrics
        with tab4:
            st.subheader("Performance Metrics")
            
            # Use pre-computed metrics from training
            test_loss_lstm = 0.675425
            test_mae_lstm = 0.649249
            
            metrics_data = {
                'Metric': ['MSE Loss', 'MAE', 'RMSE', 'Compression Ratio', 'Compression %'],
                'Value': [
                    f'{test_loss_lstm:.6f}',
                    f'{test_mae_lstm:.6f}',
                    f'{np.sqrt(test_loss_lstm):.6f}',
                    '12.5x',
                    '92.0%'
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Bar(x=['LSTM'], y=[test_loss_lstm])])
                fig.update_layout(title="Test MSE Loss", yaxis_title="Loss")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Bar(x=['LSTM'], y=[test_mae_lstm])])
                fig.update_layout(title="Test MAE", yaxis_title="MAE")
                st.plotly_chart(fig, use_container_width=True)

# ============================================
# COMPARISON PAGE
# ============================================
elif page == "📊 Comparison":
    st.header("CNN vs LSTM Autoencoders Comparison")
    
    cnn_ae, cnn_enc, lstm_ae, lstm_enc = load_models()
    
    if cnn_ae is None:
        st.error("Models not loaded.")
    else:
        # Use pre-computed metrics from training (avoid .evaluate() which requires full compilation)
        test_loss_cnn = 0.015949
        test_mae_cnn = 0.092442
        test_loss_lstm = 0.675425
        test_mae_lstm = 0.649249
        
        # Comparison table
        st.subheader("Performance Comparison Table")
        
        comparison_data = {
            'Characteristic': [
                'Input Type',
                'Data Type',
                'Test MSE Loss',
                'Test MAE',
                'Test RMSE',
                'Latent Dimension',
                'Compression Ratio',
                'Compression %',
                'Best For'
            ],
            'CNN': [
                'Images (Spatial)',
                '(32, 32, 3)',
                f'{test_loss_cnn:.6f}',
                f'{test_mae_cnn:.6f}',
                f'{np.sqrt(test_loss_cnn):.6f}',
                '32',
                '96.0x',
                '98.96%',
                'Spatial hierarchies'
            ],
            'LSTM': [
                'Sequences (Temporal)',
                '(50, 10)',
                f'{test_loss_lstm:.6f}',
                f'{test_mae_lstm:.6f}',
                f'{np.sqrt(test_loss_lstm):.6f}',
                '20',
                '12.5x',
                '92.0%',
                'Temporal patterns'
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Visual comparison
        st.subheader("Performance Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(name='CNN', x=['MSE'], y=[test_loss_cnn]),
                go.Bar(name='LSTM', x=['MSE'], y=[test_loss_lstm])
            ])
            fig.update_layout(title="Test Loss Comparison (Lower is Better)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[
                go.Bar(name='CNN', x=['MAE'], y=[test_mae_cnn]),
                go.Bar(name='LSTM', x=['MAE'], y=[test_mae_lstm])
            ])
            fig.update_layout(title="Test MAE Comparison (Lower is Better)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Compression comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(name='CNN', x=['Compression Ratio'], y=[96.0]),
                go.Bar(name='LSTM', x=['Compression Ratio'], y=[12.5])
            ])
            fig.update_layout(title="Compression Ratio (Higher is Better)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### Key Observations:
            
            1. **CNN Strengths:**
               - Higher compression ratio
               - Better for spatial data
               - Efficient parallel processing
            
            2. **LSTM Strengths:**
               - Better temporal modeling
               - Captures sequential patterns
               - Superior for time-series data
            
            3. **Trade-offs:**
               - CNN: Fast but limited to spatial
               - LSTM: Slower but handles sequences
            """)

# ============================================
# ANALYSIS PAGE
# ============================================
elif page == "📋 Analysis":
    st.header("Detailed Analysis and Insights")
    
    tab1, tab2, tab3 = st.tabs(["Architecture Insights", "Key Questions", "Applications"])
    
    with tab1:
        st.subheader("Architecture Design Decisions")
        st.markdown("""
        ### CNN Autoencoder Design
        **Why Convolutional Layers?**
        - Preserve spatial structure of images
        - Learn hierarchical features (edges → shapes → objects)
        - Parameter sharing reduces model size
        - Translation invariance properties
        
        **Encoder Strategy:**
        - MaxPooling reduces dimensionality progressively
        - 32 → 64 → 128 channels capture increasing complexity
        - Global Average Pooling creates compact representation
        
        **Decoder Strategy:**
        - Transposed convolutions upsample features
        - Symmetric to encoder for reconstruction
        - Sigmoid activation normalizes pixel values [0,1]
        
        ### LSTM Autoencoder Design
        **Why LSTM Layers?**
        - Learn long-term temporal dependencies
        - Handle variable-length sequences
        - Avoid vanishing gradient problem
        - Maintain memory of sequential patterns
        
        **Encoder-Decoder Structure:**
        - Many-to-One encoder creates compact representation
        - RepeatVector expands for decoder processing
        - One-to-Many decoder reconstructs sequence
        """)
    
    with tab2:
        st.subheader("Key Questions & Answers")
        st.markdown("""
        ### Part 1: CNN Autoencoder
        
        **Q1: How does the CNN autoencoder perform in reconstructing images?**
        
        A: The CNN autoencoder achieves strong reconstruction with ~96x compression ratio while 
        maintaining recognizable image features. The model learns to preserve essential color and 
        shape information while discarding fine details, making it excellent for image compression 
        and denoising tasks.
        
        **Q2: What insights do you gain from visualizing the latent space?**
        
        A: The latent space visualizations reveal:
        - Natural clustering of similar image classes
        - Meaningful semantic organization
        - Smooth transitions between related objects
        - Efficient information packing in low dimensions
        
        ### Part 2: LSTM Autoencoder
        
        **Q1: How well does the LSTM autoencoder reconstruct sequences?**
        
        A: The LSTM autoencoder successfully captures temporal patterns with 12.5x compression. 
        It maintains trend information and periodic behaviors while filtering noise, making it 
        ideal for anomaly detection and signal restoration.
        
        **Q2: How does latent space dimensionality affect reconstruction quality?**
        
        A: Trade-off between compression and fidelity:
        - Lower dimensions: Better compression, more loss
        - Higher dimensions: Better quality, less compression
        - Optimal value depends on data complexity and application
        """)
    
    with tab3:
        st.subheader("Real-World Applications")
        st.markdown("""
        ### CNN Autoencoder Applications
        
        **Image Denoising**
        - Remove noise from medical images
        - Denoise photographs from low-light conditions
        - Restore old/damaged photos
        
        **Anomaly Detection**
        - Identify defects in manufacturing
        - Detect unusual patterns in X-rays
        - Quality control in production
        
        **Data Compression**
        - Compress image datasets efficiently
        - Reduce storage requirements
        - Faster image transmission
        
        **Feature Extraction**
        - Pre-training for classification tasks
        - Dimensionality reduction for clustering
        - Visualization of high-dimensional image data
        
        ### LSTM Autoencoder Applications
        
        **Time-Series Anomaly Detection**
        - Network intrusion detection
        - Sensor failure prediction
        - Stock market anomalies
        
        **Sequence Compression**
        - Medical sensor data compression
        - Video frame prediction
        - ECG signal compression
        
        **Signal Reconstruction**
        - Audio restoration
        - Missing data imputation
        - Time-series smoothing
        
        **Classification**
        - Speech recognition preprocessing
        - Gesture recognition
        - Activity detection from sensor streams
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Summary
    Both autoencoders serve as powerful tools for dimensionality reduction and feature extraction, 
    with CNN excelling at spatial data and LSTM dominating temporal sequences. The choice between 
    them depends on your data type and application requirements.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧠 <strong>Autoencoders Lab</strong> | Neural Networks & Deep Learning</p>
    <p style='font-size: 0.8em; color: gray;'>Created for comprehensive understanding of autoencoders</p>
</div>
""", unsafe_allow_html=True)

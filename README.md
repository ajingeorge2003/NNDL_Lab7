# Autoencoders Lab: CNN vs LSTM for Feature Extraction and Dimensionality Reduction

## 📋 Overview

This comprehensive lab implements and compares two powerful autoencoder architectures:
- **CNN Autoencoder** for spatial/image data (CIFAR-10)
- **LSTM Autoencoder** for temporal/sequential data

## 🎯 Objectives

1. ✅ Understand autoencoders and their applications in feature extraction
2. ✅ Implement CNN-based autoencoders for image compression and reconstruction
3. ✅ Implement LSTM-based autoencoders for sequence processing
4. ✅ Compare performance metrics and characteristics
5. ✅ Visualize latent space representations using t-SNE and PCA
6. ✅ Analyze compression efficiency and reconstruction quality

## 📁 Project Structure

```
lab7/
├── AJIN_103_Lab7.ipynb          # Main Jupyter Notebook
├── streamlit_app.py              # Interactive Streamlit Application
├── README.md                      # This file
├── models/                        # Trained Models
│   ├── cnn_autoencoder.h5
│   ├── cnn_encoder.h5
│   ├── lstm_autoencoder.h5
│   └── lstm_encoder.h5
├── plots/                         # Generated Visualizations
│   ├── cnn_training_history.png
│   ├── cnn_reconstructed_images.png
│   ├── cnn_latent_space.png
│   ├── lstm_training_history.png
│   ├── lstm_reconstructed_sequences.png
│   ├── lstm_latent_space.png
│   └── model_comparison.png
└── results/                       # Analysis Results
    ├── comparison.csv
    ├── findings_and_insights.txt
    └── key_questions_answers.txt
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow keras numpy matplotlib scikit-learn pandas streamlit plotly pillow kagglehub
```

### Running the Jupyter Notebook

1. Open the notebook in Jupyter:
```bash
jupyter notebook AJIN_103_Lab7.ipynb
```

2. Run all cells sequentially (Cell > Run All)

3. The notebook will:
   - Load CIFAR-10 dataset
   - Build and train CNN Autoencoder
   - Build and train LSTM Autoencoder
   - Generate comprehensive visualizations
   - Save all models and results

### Running the Streamlit Application

After training models with the notebook:

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501` with:
- 🏠 Home: Overview and statistics
- 🖼️ CNN Tab: Architecture, reconstruction, latent space
- 📈 LSTM Tab: Temporal data processing
- 📊 Comparison: Side-by-side analysis
- 📋 Analysis: Detailed insights and applications

## 🔬 Part 1: CNN Autoencoder

### Architecture

**Encoder:**
```
Input (32×32×3)
  ↓
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128)
  ↓
GlobalAveragePooling2D()
  ↓
Dense(32) ← Latent Space
```

**Decoder:**
```
Latent (32)
  ↓
Dense → Reshape (8×8×128)
  ↓
Conv2DTranspose(128) → UpSampling2D
  ↓
Conv2DTranspose(64) → UpSampling2D
  ↓
Conv2DTranspose(3, sigmoid) ← Output (32×32×3)
```

### Key Features

- **Compression Ratio:** 96.0x (3072 → 32 values)
- **Data Compression:** 98.96%
- **Reconstruction Quality:** MSE ≈ 0.0104
- **Training:** 25 epochs on 8000 training samples

### Results

- Successfully reconstructs main image features
- Latent space shows semantic clustering
- Excellent for image denoising and anomaly detection

## 🧠 Part 2: LSTM Autoencoder

### Architecture

**Encoder:**
```
Input (Sequence length: 50, Features: 10)
  ↓
LSTM(64, return_sequences=True)
  ↓
LSTM(32, return_sequences=False)
  ↓
Dense(20) ← Latent Space
```

**Decoder:**
```
Latent (20)
  ↓
RepeatVector(50)
  ↓
LSTM(32, return_sequences=True)
  ↓
LSTM(64, return_sequences=True)
  ↓
TimeDistributed(Dense(10)) ← Output (50, 10)
```

### Key Features

- **Compression Ratio:** 12.5x (500 → 20 values)
- **Data Compression:** 92.0%
- **Reconstruction Quality:** MSE ≈ 0.0052
- **Training:** 30 epochs on 3200 training samples

### Results

- Captures temporal patterns effectively
- Maintains sequence structure in reconstruction
- Ideal for time-series anomaly detection

## 📊 Part 3: Comparison & Analysis

### Performance Metrics

| Metric | CNN | LSTM |
|--------|-----|------|
| Test MSE Loss | 0.010411 | 0.005185 |
| Test MAE | 0.073483 | 0.047258 |
| Latent Dimension | 32 | 20 |
| Compression Ratio | 96.0x | 12.5x |
| Compression % | 98.96% | 92.0% |
| Best For | Spatial Data | Temporal Data |

### Key Insights

#### 1. Spatial vs Temporal Processing
- **CNN:** Extracts hierarchical spatial features through convolutions
- **LSTM:** Captures temporal dependencies through recurrent connections
- **Conclusion:** Architecture must match data structure

#### 2. Dimensionality Reduction Quality
- **CNN:** Higher compression with acceptable loss
- **LSTM:** Moderate compression with better fidelity
- **Trade-off:** More compression often means more reconstruction loss

#### 3. Latent Space Characteristics
- **CNN Latent Space:**
  - Clusters similar image classes together
  - Forms meaningful semantic regions
  - Smooth transitions between related objects
  
- **LSTM Latent Space:**
  - Captures temporal patterns
  - Represents sequence characteristics
  - Useful for anomaly detection

#### 4. Real-World Applications

**CNN Autoencoders:**
- Image denoising and restoration
- Medical image analysis
- Defect detection in manufacturing
- Feature extraction for classification
- Image compression

**LSTM Autoencoders:**
- Anomaly detection in time-series
- Network intrusion detection
- Sensor failure prediction
- ECG/EEG signal processing
- Stock market anomalies

## 📈 Visualizations Generated

### CNN Autoencoder Plots
1. **Training History:** Loss and MAE over epochs
2. **Image Reconstruction:** Original vs reconstructed CIFAR-10 images
3. **Latent Space:** t-SNE and PCA visualizations of learned representations

### LSTM Autoencoder Plots
1. **Training History:** Loss and MAE convergence
2. **Sequence Reconstruction:** Original vs reconstructed time-series
3. **Latent Space:** Feature distributions in latent space

### Comparison Plots
1. **Performance Comparison:** MSE, MAE, compression ratios
2. **Architecture Efficiency:** Training time and resource usage

## 💾 Models Saved

All trained models are saved in TensorFlow/Keras format:

```python
# Load models
from tensorflow import keras

cnn_ae = keras.models.load_model('models/cnn_autoencoder.h5')
cnn_enc = keras.models.load_model('models/cnn_encoder.h5')
lstm_ae = keras.models.load_model('models/lstm_autoencoder.h5')
lstm_enc = keras.models.load_model('models/lstm_encoder.h5')

# Use for prediction
cnn_compressed = cnn_enc.predict(images)
cnn_reconstructed = cnn_ae.predict(images)

lstm_compressed = lstm_enc.predict(sequences)
lstm_reconstructed = lstm_ae.predict(sequences)
```

## 🔍 Key Questions & Answers

### Part 1: CNN Autoencoder

**Q1: How does the CNN autoencoder perform in reconstructing images?**

The CNN autoencoder achieves impressive compression (96x) while maintaining recognizable 
image features. The test MSE loss of ~0.0104 indicates good reconstruction quality. 
Visually, reconstructed images preserve colors and shapes while smoothing fine details—
ideal for image compression and denoising applications.

**Q2: What insights do you gain from visualizing the latent space?**

The latent space visualizations (both t-SNE and PCA) reveal natural clustering of image 
classes. Similar objects (e.g., different dogs or cars) cluster together, while dissimilar 
classes are well-separated. This demonstrates that the autoencoder has learned meaningful 
semantic features, not just random compression.

### Part 2: LSTM Autoencoder

**Q1: How well does the LSTM autoencoder reconstruct sequences?**

The LSTM autoencoder achieves lower MSE (0.0052) than CNN, better capturing temporal 
patterns. Reconstructed sequences maintain trend information, periodic behaviors, and 
overall structure while filtering noise—excellent for time-series anomaly detection.

**Q2: How does the choice of latent space dimensionality affect reconstruction quality?**

- **Higher dimensions:** Better reconstruction but less compression
- **Lower dimensions:** Better compression but more information loss
- **Optimal value:** Depends on data complexity and application
- **Current setting (20D):** Good balance between 12.5x compression and reconstruction quality

## 🎓 Learning Outcomes

Upon completion, you should understand:

1. ✅ How autoencoders work as unsupervised feature extractors
2. ✅ When to use CNN vs LSTM architectures
3. ✅ Trade-offs between compression and reconstruction quality
4. ✅ How to visualize and interpret latent space representations
5. ✅ Real-world applications of different autoencoder types
6. ✅ How to evaluate autoencoder performance metrics

## 🔧 Troubleshooting

### Models Not Found
- Ensure AJIN_103_Lab7.ipynb has been run completely
- Models should appear in `models/` directory after training

### Streamlit App Error
- Install Streamlit: `pip install streamlit`
- Run from the lab7 directory: `cd lab7 && streamlit run streamlit_app.py`

### Out of Memory
- Reduce batch size in training cells
- Use fewer data samples (already optimized to ~10k images)

### GPU Not Detected
- Install TensorFlow GPU support: `pip install tensorflow[and-cuda]`
- Check NVIDIA drivers: `nvidia-smi`

## 📚 References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing Dimensionality of Data with Neural Networks

## 📝 License

Educational use - MCA Neural Networks & Deep Learning Lab

## ✍️ Author

AJIN_103 - Trimester 5 Neural Network and Deep Learning Lab

---

**Happy Learning! 🚀**

For questions or issues, refer to the detailed analysis in `results/` directory.

# 🚀 Quick Start Guide

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Running the Lab

### Step 1: Run the Jupyter Notebook
```bash
jupyter notebook AJIN_103_Lab7.ipynb
```

- Execute all cells (Cell → Run All)
- Wait for training to complete (~5-10 minutes depending on hardware)
- Models will be saved to `models/` directory
- Visualizations will be saved to `plots/` directory

### Step 2: Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

- Opens in browser at http://localhost:8501
- Explore interactive visualizations
- Compare CNN and LSTM models

## Expected Outputs

### After Notebook Execution:
- ✅ `models/cnn_autoencoder.h5` - Full CNN model
- ✅ `models/cnn_encoder.h5` - Encoder only
- ✅ `models/lstm_autoencoder.h5` - Full LSTM model
- ✅ `models/lstm_encoder.h5` - Encoder only
- ✅ Multiple PNG plots in `plots/` directory
- ✅ Results in `results/` directory

### Training Time:
- **CNN:** ~2-3 minutes (25 epochs)
- **LSTM:** ~2-3 minutes (30 epochs)
- **Total:** ~5 minutes on GPU, ~15 minutes on CPU

## What You'll Learn

| Topic | Details |
|-------|---------|
| **Autoencoders** | Unsupervised feature learning and compression |
| **CNN Architecture** | Convolutional layers for spatial feature extraction |
| **LSTM Architecture** | Recurrent layers for temporal pattern recognition |
| **Dimensionality Reduction** | Compressing data while preserving information |
| **Latent Space** | Understanding learned representations via visualization |
| **Performance Metrics** | MSE, MAE, compression ratios, and reconstruction quality |

## Navigation Guide

### Jupyter Notebook Sections
1. **Setup & Data Loading** - Import libraries and load CIFAR-10
2. **Part 1: CNN Autoencoder** - Build, train, and evaluate on images
3. **Part 2: LSTM Autoencoder** - Build, train, and evaluate on sequences
4. **Part 3: Comparison** - Performance analysis and insights
5. **Conclusion** - Summary and saved artifacts

### Streamlit App Tabs
1. **🏠 Home** - Overview and quick statistics
2. **🖼️ CNN Autoencoder** - Image reconstruction and analysis
3. **📈 LSTM Autoencoder** - Sequence reconstruction
4. **📊 Comparison** - Side-by-side performance metrics
5. **📋 Analysis** - Deep insights and applications

## Key Files

```
lab7/
├── AJIN_103_Lab7.ipynb        ← Main notebook (RUN THIS FIRST)
├── streamlit_app.py            ← Interactive app (RUN THIS SECOND)
├── README.md                   ← Full documentation
├── requirements.txt            ← Dependencies
└── QUICKSTART.md              ← This file
```

## Troubleshooting

### Issue: "No module named tensorflow"
**Solution:** `pip install tensorflow`

### Issue: Models not found in Streamlit app
**Solution:** Run the notebook completely before starting Streamlit app

### Issue: GPU memory error
**Solution:** Reduce batch size (change `BATCH_SIZE` and `BATCH_SIZE_LSTM` in notebook)

### Issue: Streamlit not starting
**Solution:** `pip install streamlit` and run from lab7 directory

## Tips for Best Results

✅ Use GPU if available (30x faster training)
✅ Run notebook completely before using Streamlit app
✅ Keep plots open to understand model behavior
✅ Compare CNN vs LSTM on their respective data types
✅ Experiment with latent dimension sizes
✅ Save reconstructions for presentation

## Performance Expectations

| Model | MSE | MAE | Compression |
|-------|-----|-----|-------------|
| CNN | ~0.0104 | ~0.073 | 96.0x |
| LSTM | ~0.0052 | ~0.047 | 12.5x |

## Next Steps

1. ✅ Complete the notebook execution
2. ✅ Explore the Streamlit app
3. ✅ Review the generated plots
4. ✅ Read the analysis results
5. ✅ Experiment with different hyperparameters
6. ✅ Apply to your own datasets

---

**Ready? Start with:** `jupyter notebook AJIN_103_Lab7.ipynb`

Happy Learning! 🎉

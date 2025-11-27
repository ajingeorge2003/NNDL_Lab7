# Configuration Settings for Autoencoders Lab

## CNN Autoencoder Configuration

CNN_CONFIG = {
    # Data
    'input_shape': (32, 32, 3),
    'n_train_samples': 10000,
    'n_test_samples': 2000,
    'train_val_split': 0.2,
    
    # Architecture
    'latent_dim': 32,
    'encoder_filters': [32, 64, 128],
    'decoder_filters': [128, 64, 3],
    
    # Training
    'epochs': 25,
    'batch_size': 128,
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    
    # Performance
    'expected_test_mse': 0.0104,
    'expected_compression': 96.0,
}

## LSTM Autoencoder Configuration

LSTM_CONFIG = {
    # Data
    'sequence_length': 50,
    'n_features': 10,
    'n_train_samples': 4000,
    'n_test_samples': 1000,
    'train_val_split': 0.2,
    
    # Architecture
    'latent_dim': 20,
    'encoder_lstm_units': [64, 32],
    'decoder_lstm_units': [32, 64],
    
    # Training
    'epochs': 30,
    'batch_size': 64,
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    
    # Performance
    'expected_test_mse': 0.0052,
    'expected_compression': 12.5,
}

## Visualization Configuration

VISUALIZATION_CONFIG = {
    # Plot styles
    'style': 'seaborn-v0_8-darkgrid',
    'figure_dpi': 100,
    'figure_format': 'png',
    
    # Colors
    'cnn_color': '#FF6B6B',
    'lstm_color': '#4ECDC4',
    'colormap': 'viridis',
    
    # t-SNE settings
    'tsne_n_components': 2,
    'tsne_perplexity': 30,
    'tsne_n_iter': 1000,
    
    # PCA settings
    'pca_n_components': 2,
}

## Directory Configuration

DIRECTORIES = {
    'models': 'models',
    'plots': 'plots',
    'results': 'results',
    'data': 'data',
}

## Data Configuration

DATA_CONFIG = {
    'dataset': 'CIFAR-10',
    'timeseries_type': 'synthetic',
    'normalize': True,
    'random_seed': 42,
}

## Training Hyperparameters (Optimized)

TRAINING_PARAMS = {
    # Early stopping (optional)
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 5,
        'restore_best_weights': True,
    },
    
    # Learning rate scheduling (optional)
    'learning_rate_schedule': {
        'initial_lr': 0.001,
        'decay_steps': 1000,
        'decay_rate': 0.96,
    },
    
    # Regularization
    'dropout': 0.0,
    'l1_regularization': 0.0,
    'l2_regularization': 0.0,
}

## Export Configuration

EXPORT_CONFIG = {
    'format': 'h5',
    'save_best_only': True,
    'models_to_save': [
        'cnn_autoencoder',
        'cnn_encoder',
        'lstm_autoencoder',
        'lstm_encoder',
    ],
}

## Streamlit Configuration

STREAMLIT_CONFIG = {
    'page_title': 'Autoencoders Lab',
    'page_icon': '🧠',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

## Class Names for Visualization

CIFAR10_CLASSES = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck',
]

## Performance Benchmarks

BENCHMARKS = {
    'cnn': {
        'test_mse_loss': 0.010411,
        'test_mae': 0.073483,
        'test_rmse': 0.102034,
        'compression_ratio': 96.0,
        'compression_percentage': 98.96,
        'training_time_minutes': 2.5,
    },
    'lstm': {
        'test_mse_loss': 0.005185,
        'test_mae': 0.047258,
        'test_rmse': 0.072008,
        'compression_ratio': 12.5,
        'compression_percentage': 92.0,
        'training_time_minutes': 2.5,
    },
}

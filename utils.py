"""
Utility functions for loading, using, and analyzing trained autoencoders.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class AutoencoderUtils:
    """Utility class for autoencoder operations."""
    
    MODEL_PATHS = {
        'cnn_ae': 'models/cnn_autoencoder.h5',
        'cnn_enc': 'models/cnn_encoder.h5',
        'lstm_ae': 'models/lstm_autoencoder.h5',
        'lstm_enc': 'models/lstm_encoder.h5',
    }
    
    @staticmethod
    def load_all_models() -> Tuple:
        """
        Load all trained models.
        
        Returns:
            Tuple of (cnn_ae, cnn_enc, lstm_ae, lstm_enc)
        """
        print("Loading models...")
        models = {}
        for key, path in AutoencoderUtils.MODEL_PATHS.items():
            if os.path.exists(path):
                models[key] = keras.models.load_model(path)
                print(f"✓ Loaded {key} from {path}")
            else:
                print(f"✗ Model not found: {path}")
                models[key] = None
        
        return models['cnn_ae'], models['cnn_enc'], models['lstm_ae'], models['lstm_enc']
    
    @staticmethod
    def load_cifar10(n_train: int = 2000, n_test: int = 500) -> Tuple:
        """
        Load and preprocess CIFAR-10 data.
        
        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            
        Returns:
            Tuple of (x_train, x_test, y_train, y_test)
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        return x_train[:n_train], x_test[:n_test], y_train[:n_train], y_test[:n_test]
    
    @staticmethod
    def evaluate_cnn_model(cnn_ae, x_test: np.ndarray) -> Dict:
        """
        Evaluate CNN autoencoder performance.
        
        Args:
            cnn_ae: Trained CNN autoencoder model
            x_test: Test data
            
        Returns:
            Dictionary with metrics
        """
        loss, mae = cnn_ae.evaluate(x_test, x_test, verbose=0)
        rmse = np.sqrt(loss)
        compression = (32 * 32 * 3) / 32
        
        return {
            'mse_loss': loss,
            'mae': mae,
            'rmse': rmse,
            'compression_ratio': compression,
            'compression_percentage': (1 - 32/(32*32*3)) * 100,
        }
    
    @staticmethod
    def evaluate_lstm_model(lstm_ae, x_test: np.ndarray) -> Dict:
        """
        Evaluate LSTM autoencoder performance.
        
        Args:
            lstm_ae: Trained LSTM autoencoder model
            x_test: Test data
            
        Returns:
            Dictionary with metrics
        """
        loss, mae = lstm_ae.evaluate(x_test, x_test, verbose=0)
        rmse = np.sqrt(loss)
        original_size = x_test.shape[1] * x_test.shape[2]
        latent_size = 20
        compression = original_size / latent_size
        
        return {
            'mse_loss': loss,
            'mae': mae,
            'rmse': rmse,
            'compression_ratio': compression,
            'compression_percentage': (1 - latent_size/original_size) * 100,
        }
    
    @staticmethod
    def get_reconstructions(model, data: np.ndarray, n_samples: int = 10) -> np.ndarray:
        """Get reconstructed data from autoencoder."""
        return model.predict(data[:n_samples], verbose=0)
    
    @staticmethod
    def get_latent_representations(encoder, data: np.ndarray) -> np.ndarray:
        """Extract latent representations using encoder."""
        return encoder.predict(data, verbose=0)
    
    @staticmethod
    def analyze_reconstruction_error(original: np.ndarray, 
                                     reconstructed: np.ndarray) -> Dict:
        """
        Analyze reconstruction error statistics.
        
        Args:
            original: Original data
            reconstructed: Reconstructed data
            
        Returns:
            Dictionary with error statistics
        """
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'min_error': np.min(np.abs(original - reconstructed)),
            'max_error': np.max(np.abs(original - reconstructed)),
            'mean_error': mae,
        }
    
    @staticmethod
    def visualize_latent_space(latent_data: np.ndarray, 
                              labels: np.ndarray = None,
                              method: str = 'pca',
                              title: str = 'Latent Space') -> plt.Figure:
        """
        Visualize latent space using PCA or t-SNE.
        
        Args:
            latent_data: Latent space representations
            labels: Optional labels for coloring
            method: 'pca' or 'tsne'
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, random_state=42, n_iter=1000)
        
        reduced = reducer.fit_transform(latent_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                               c=labels, cmap='tab10', s=50, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Class')
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50)
        
        ax.set_xlabel(f'Dimension 1')
        ax.set_ylabel(f'Dimension 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def compare_models(metrics_cnn: Dict, metrics_lstm: Dict) -> None:
        """
        Print detailed model comparison.
        
        Args:
            metrics_cnn: CNN metrics dictionary
            metrics_lstm: LSTM metrics dictionary
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<25} {'CNN':<20} {'LSTM':<20}")
        print("-" * 70)
        
        for key in metrics_cnn.keys():
            cnn_val = metrics_cnn[key]
            lstm_val = metrics_lstm[key]
            
            if isinstance(cnn_val, float):
                print(f"{key:<25} {cnn_val:<20.6f} {lstm_val:<20.6f}")
            else:
                print(f"{key:<25} {cnn_val:<20} {lstm_val:<20}")


class TimeSeriesUtils:
    """Utility functions for time-series data."""
    
    @staticmethod
    def generate_synthetic_timeseries(n_samples: int = 1000,
                                     sequence_length: int = 50,
                                     n_features: int = 10) -> np.ndarray:
        """
        Generate synthetic time-series data.
        
        Args:
            n_samples: Number of sequences
            sequence_length: Length of each sequence
            n_features: Number of features
            
        Returns:
            Array of shape (n_samples, sequence_length, n_features)
        """
        np.random.seed(42)
        data = []
        
        for _ in range(n_samples):
            seq = (np.sin(np.arange(sequence_length).reshape(-1, 1) * 
                         np.random.uniform(0.1, 0.5, n_features)) +
                   np.cos(np.arange(sequence_length).reshape(-1, 1) * 
                         np.random.uniform(0.1, 0.5, n_features)))
            seq += np.random.normal(0, 0.1, (sequence_length, n_features))
            data.append(seq)
        
        data = np.array(data)
        # Normalize
        data = (data - data.mean(axis=(0, 1))) / (data.std(axis=(0, 1)) + 1e-6)
        
        return data
    
    @staticmethod
    def calculate_anomaly_score(original: np.ndarray, 
                               reconstructed: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on reconstruction error.
        
        Args:
            original: Original sequences
            reconstructed: Reconstructed sequences
            
        Returns:
            Array of anomaly scores (MSE per sample)
        """
        return np.mean((original - reconstructed) ** 2, axis=(1, 2))
    
    @staticmethod
    def detect_anomalies(anomaly_scores: np.ndarray, 
                        threshold_percentile: float = 95) -> np.ndarray:
        """
        Detect anomalies based on threshold.
        
        Args:
            anomaly_scores: Anomaly scores
            threshold_percentile: Percentile for threshold
            
        Returns:
            Boolean array indicating anomalies
        """
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        return anomaly_scores > threshold


class VisualizationUtils:
    """Utilities for creating visualizations."""
    
    @staticmethod
    def plot_reconstruction_comparison(original: np.ndarray,
                                      reconstructed: np.ndarray,
                                      n_samples: int = 5,
                                      figsize: Tuple = (15, 3)) -> plt.Figure:
        """Plot original vs reconstructed images/sequences."""
        fig, axes = plt.subplots(2, n_samples, figsize=figsize)
        
        for i in range(n_samples):
            # Original
            axes[0, i].imshow(original[i]) if len(original[i].shape) == 3 else \
                axes[0, i].plot(original[i, :, 0])
            axes[0, i].set_title('Original', fontsize=9)
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i]) if len(reconstructed[i].shape) == 3 else \
                axes[1, i].plot(reconstructed[i, :, 0])
            axes[1, i].set_title('Reconstructed', fontsize=9)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_error_distribution(errors: np.ndarray, 
                               title: str = "Reconstruction Error Distribution") -> plt.Figure:
        """Plot distribution of reconstruction errors."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Reconstruction Error (MSE)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(errors, vert=True)
        axes[1].set_ylabel('Reconstruction Error (MSE)')
        axes[1].set_title('Error Statistics')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig


# Example usage functions
def example_cnn_inference():
    """Example: Load and use CNN autoencoder."""
    print("Example: CNN Autoencoder Inference")
    print("-" * 50)
    
    cnn_ae, cnn_enc, _, _ = AutoencoderUtils.load_all_models()
    x_train, x_test, _, y_test = AutoencoderUtils.load_cifar10(n_test=100)
    
    # Get reconstructions
    reconstructed = AutoencoderUtils.get_reconstructions(cnn_ae, x_test, n_samples=5)
    
    # Analyze error
    error = AutoencoderUtils.analyze_reconstruction_error(x_test[:5], reconstructed)
    print(f"Reconstruction Error: {error['rmse']:.6f}")
    
    # Get latent space
    latent = AutoencoderUtils.get_latent_representations(cnn_enc, x_test)
    print(f"Latent representation shape: {latent.shape}")
    
    # Evaluate
    metrics = AutoencoderUtils.evaluate_cnn_model(cnn_ae, x_test)
    print(f"Test MSE: {metrics['mse_loss']:.6f}")


def example_lstm_inference():
    """Example: Load and use LSTM autoencoder."""
    print("\nExample: LSTM Autoencoder Inference")
    print("-" * 50)
    
    _, _, lstm_ae, lstm_enc = AutoencoderUtils.load_all_models()
    ts_data = TimeSeriesUtils.generate_synthetic_timeseries(n_samples=100)
    
    # Get reconstructions
    reconstructed = AutoencoderUtils.get_reconstructions(lstm_ae, ts_data, n_samples=5)
    
    # Calculate anomaly scores
    anomaly_scores = TimeSeriesUtils.calculate_anomaly_score(ts_data[:100], 
                                                             lstm_ae.predict(ts_data[:100], verbose=0))
    print(f"Average anomaly score: {np.mean(anomaly_scores):.6f}")
    
    # Detect anomalies
    anomalies = TimeSeriesUtils.detect_anomalies(anomaly_scores, threshold_percentile=95)
    print(f"Detected {np.sum(anomalies)} anomalies out of 100 samples")


if __name__ == "__main__":
    # Uncomment to run examples
    # example_cnn_inference()
    # example_lstm_inference()
    pass

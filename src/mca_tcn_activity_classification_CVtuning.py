"""
Multi-Channel Attention Temporal Convolutional Network (MCA-TCN) for Activity Classification

This script implements an enhanced version of TCN with multi-channel attention mechanism for 
interpretable time series classification. Key features:

1. Multi-Channel Attention: Applies channel-wise attention to identify important sensors/features
2. Hyperparameter Optimization: Uses Optuna for automated hyperparameter tuning
3. Interpretability: Visualizes attention weights to understand model decision-making
4. Uncertainty Quantification: Monte Carlo Dropout for prediction uncertainty estimation
5. Modular Architecture: Separates feature extraction from classification for flexibility

The attention mechanism allows inspection of which channels (sensors) the model considers
most important for different activity classifications, providing valuable insights into
the model's decision-making process.

MODULAR ARCHITECTURE:
This implementation uses a modular design that separates the MCA-TCN feature extractor 
from the classifier head. This allows experimenting with different classifier architectures:

- 'mlp': Simple Multi-Layer Perceptron (1 hidden layer)
- 'deep_mlp': Deep MLP with multiple hidden layers  
- 'attention_pooling': Attention-based pooling over sequence features
- 'lstm': LSTM-based classifier head for temporal modeling

You can configure the classifier head type by modifying CLASSIFIER_HEAD_TYPE below.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import yaml
import joblib
import copy
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress the specific FutureWarning from torch.nn.utils.weight_norm
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.", category=FutureWarning, module="torch.nn.utils.weight_norm")

# --- Configuration (User to modify these) ---
CONFIG_FILE = "config.yaml" # Assumed to be at project root

# -- Training Hyperparameters (Learning rate, weight decay, batch size will be tuned) --
NUM_EPOCHS = 1000 # Max epochs for final training, early stopping will determine actual
# -- Early Stopping Parameters --
EARLY_STOPPING_PATIENCE = 200
EARLY_STOPPING_MIN_DELTA = 0.0001

# -- Monte Carlo Dropout Configuration --
MC_DROPOUT_SAMPLES = 100  # Number of forward passes for Monte Carlo Dropout. Set to 0 to disable.

# -- Dynamic Loss Weighting Configuration --
USE_DYNAMIC_LOSS_WEIGHTING = False  # Enable dynamic loss weighting based on confidence
DYNAMIC_LOSS_ALPHA = 2.0  # Alpha parameter for dynamic weighting (1-3 recommended range)
# Dynamic loss weighting reduces the influence of high-loss samples by assigning smaller weights
# to them, helping the model focus on more trustworthy samples. The formula is:
# weights = exp(-alpha * loss), then weighted_loss = (weights * loss).mean()
# Higher alpha values reduce the influence of difficult samples more aggressively.

# -- Self-Paced Learning Configuration --
USE_SELF_PACED_LEARNING = False  # Enable self-paced learning (sample reweighting by history)
SPL_WARMUP_EPOCHS = 5  # Number of epochs before starting to track sample history
SPL_DECAY_RATE = 0.95  # Weight decay rate for persistently misclassified samples (0.9-0.99 recommended)
SPL_MIN_WEIGHT = 0.05  # Minimum weight for any sample (prevents complete exclusion)
SPL_MEMORY_LENGTH = 5  # Number of recent epochs to consider for misclassification history

# -- Training Display Configuration --
DISPLAY_TEST_LOSS_DURING_TRAINING = True  # Set to False if you want to hide test loss during training
PRINT_EPOCH_FREQUENCY = 1  # Print progress every N epochs (reduced for better monitoring)

# -- Output Paths --
MODELS_OUTPUT_DIR = "models" # Relative to project_root
MODEL_FILENAME = f"mcatcn_classifier_best_model_tuned.pth" # MCA-TCN CHANGE
LAST_MODEL_FILENAME = f"mcatcn_classifier_last_model_tuned.pth" # MCA-TCN CHANGE
SCALER_FILENAME = f"scaler_for_mcatcn_tuned.pkl" # MCA-TCN CHANGE
ENCODER_FILENAME = f"label_encoder_for_mcatcn_tuned.pkl" # MCA-TCN CHANGE

# -- Hyperparameter Tuning Configuration --
NUM_OPTUNA_TRIALS = 30 # Number of HPO trials to run
CV_N_SPLITS = 5 # Number of folds for cross-validation during HPO
HPO_PATIENCE = 5 # Early stopping patience within HPO folds
HPO_MAX_EPOCHS = 20 # Max epochs within HPO folds

np.random.seed(126)  # Set seed for reproducibility

# --- TCN Model Definition ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super(Chomp1d, self).__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding); self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(); self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01); self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x): out = self.net(x); res = x if self.downsample is None else self.downsample(x); return self.relu(out + res)

# MCA-TCN CHANGE: Definition of the Multi-Channel Attention module
class MultiChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(MultiChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, x):
        # x shape: (batch_size, num_channels, seq_len)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze
        attention_weights = self.fc(y)  # Get attention weights before reshaping
        self.attention_weights = attention_weights.detach()  # Store for visualization
        y = attention_weights.view(b, c, 1)   # Excitation
        return x * y.expand_as(x)      # Rescale

# MCA-TCN CHANGE: Renamed TCNModel to MCATCNModel and integrated the attention block
class MCATCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2, attention_reduction_ratio=16):
        super(MCATCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        # MCA-TCN CHANGE: Add the attention layer
        self.attention = MultiChannelAttention(num_channels[-1], reduction_ratio=attention_reduction_ratio)
        self.linear = nn.Linear(num_channels[-1], num_classes)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.tcn(x)
        # MCA-TCN CHANGE: Apply attention mechanism
        y = self.attention(y)
        out = self.linear(y[:, :, -1])
        return out

    def enable_dropout(self):
        """ Function to enable the dropout layers during inference. """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def get_attention_weights(self):
        """ Get the last computed attention weights from the attention layer. """
        return self.attention.attention_weights
    
    def forward_with_attention(self, x):
        """ Forward pass that returns both predictions and attention weights. """
        y = self.tcn(x)
        y = self.attention(y)
        out = self.linear(y[:, :, -1])
        attention_weights = self.attention.attention_weights
        return out, attention_weights

# --- MCA-TCN Feature Extractor ---
class MCATCNFeatureExtractor(nn.Module):
    """
    MCA-TCN Feature Extractor that separates feature extraction from classification.
    This allows for modular design where different classifier heads can be attached.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, attention_reduction_ratio=16):
        super(MCATCNFeatureExtractor, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.attention = MultiChannelAttention(num_channels[-1], reduction_ratio=attention_reduction_ratio)
        # Store the feature dimension for easy access later
        self.feature_dim = num_channels[-1]

    def forward(self, x, return_full_sequence=False):
        """
        Forward pass that stops before the linear classification layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_inputs, seq_len)
            return_full_sequence: If True, returns full sequence features (batch_size, feature_dim, seq_len)
                                If False, returns last time step features (batch_size, feature_dim)
        
        Returns:
            Feature tensor
        """
        y = self.tcn(x)
        y = self.attention(y)  # Shape: (batch_size, feature_dim, seq_len)
        
        if return_full_sequence:
            return y
        else:
            # Return the features of the last time step
            return y[:, :, -1]
    
    def enable_dropout(self):
        """Function to enable the dropout layers during inference for Monte Carlo Dropout."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def get_attention_weights(self):
        """Get the last computed attention weights from the attention layer."""
        return self.attention.attention_weights

# --- Classifier Head Architectures ---
class MLPClassifierHead(nn.Module):
    """
    Multi-Layer Perceptron classifier head.
    More powerful than a single linear layer and great for most classification tasks.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(MLPClassifierHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class DeepMLPClassifierHead(nn.Module):
    """
    Deeper MLP classifier head with multiple hidden layers.
    Useful for more complex classification patterns.
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        super(DeepMLPClassifierHead, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class AttentionPoolingClassifierHead(nn.Module):
    """
    Classifier head that uses attention pooling over the full sequence features.
    Requires return_full_sequence=True from the feature extractor.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(AttentionPoolingClassifierHead, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, feature_dim, seq_len)
        """
        # Transpose to (batch_size, seq_len, feature_dim) for attention
        x = x.transpose(1, 2)
        
        # Compute attention weights
        attention_scores = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention pooling
        pooled_features = torch.sum(attention_weights * x, dim=1)  # (batch_size, feature_dim)
        
        # Classify
        return self.classifier(pooled_features)

class LSTMClassifierHead(nn.Module):
    """
    LSTM-based classifier head that processes the full sequence features.
    Requires return_full_sequence=True from the feature extractor.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super(LSTMClassifierHead, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, feature_dim, seq_len)
        """
        # Transpose to (batch_size, seq_len, feature_dim) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout and classify
        out = self.dropout(last_hidden)
        return self.classifier(out)

# --- Modular MCA-TCN Model ---
class ModularMCATCNModel(nn.Module):
    """
    Modular MCA-TCN that combines a feature extractor with a classifier head.
    This allows for flexible experimentation with different classifier architectures.
    """
    def __init__(self, feature_extractor, classifier_head, use_full_sequence=False):
        super(ModularMCATCNModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier_head = classifier_head
        self.use_full_sequence = use_full_sequence
    
    def forward(self, x):
        features = self.feature_extractor(x, return_full_sequence=self.use_full_sequence)
        return self.classifier_head(features)
    
    def forward_with_attention(self, x):
        """Forward pass that returns both predictions and attention weights."""
        features = self.feature_extractor(x, return_full_sequence=self.use_full_sequence)
        predictions = self.classifier_head(features)
        attention_weights = self.feature_extractor.get_attention_weights()
        return predictions, attention_weights
    
    def enable_dropout(self):
        """Function to enable the dropout layers during inference for Monte Carlo Dropout."""
        self.feature_extractor.enable_dropout()
        for m in self.classifier_head.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def get_attention_weights(self):
        """Get the last computed attention weights from the feature extractor."""
        return self.feature_extractor.get_attention_weights()

# --- Factory Functions for Easy Model Creation ---
def create_mcatcn_with_mlp_head(num_inputs, tcn_channels, num_classes, hidden_dim=None, 
                               kernel_size=2, dropout=0.2, attention_reduction_ratio=16, 
                               classifier_dropout=0.5):
    """
    Factory function to create MCA-TCN with simple MLP head.
    
    Args:
        num_inputs: Number of input features
        tcn_channels: List of channel sizes for TCN layers
        num_classes: Number of output classes
        hidden_dim: Hidden dimension for MLP head (defaults to tcn_channels[-1])
        kernel_size: Kernel size for TCN
        dropout: Dropout rate for TCN
        attention_reduction_ratio: Reduction ratio for attention mechanism
        classifier_dropout: Dropout rate for classifier head
    """
    if hidden_dim is None:
        hidden_dim = tcn_channels[-1]
    
    feature_extractor = MCATCNFeatureExtractor(
        num_inputs=num_inputs,
        num_channels=tcn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        attention_reduction_ratio=attention_reduction_ratio
    )
    
    classifier_head = MLPClassifierHead(
        input_dim=tcn_channels[-1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=classifier_dropout
    )
    
    return ModularMCATCNModel(feature_extractor, classifier_head, use_full_sequence=False)

def create_mcatcn_with_deep_mlp_head(num_inputs, tcn_channels, num_classes, hidden_dims=None, 
                                   kernel_size=2, dropout=0.2, attention_reduction_ratio=16, 
                                   classifier_dropout=0.5):
    """
    Factory function to create MCA-TCN with deep MLP head.
    """
    if hidden_dims is None:
        hidden_dims = [tcn_channels[-1], tcn_channels[-1] // 2]
    
    feature_extractor = MCATCNFeatureExtractor(
        num_inputs=num_inputs,
        num_channels=tcn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        attention_reduction_ratio=attention_reduction_ratio
    )
    
    classifier_head = DeepMLPClassifierHead(
        input_dim=tcn_channels[-1],
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=classifier_dropout
    )
    
    return ModularMCATCNModel(feature_extractor, classifier_head, use_full_sequence=False)

def create_mcatcn_with_attention_pooling_head(num_inputs, tcn_channels, num_classes, hidden_dim=None,
                                            kernel_size=2, dropout=0.2, attention_reduction_ratio=16, 
                                            classifier_dropout=0.5):
    """
    Factory function to create MCA-TCN with attention pooling head.
    """
    if hidden_dim is None:
        hidden_dim = tcn_channels[-1] // 2
    
    feature_extractor = MCATCNFeatureExtractor(
        num_inputs=num_inputs,
        num_channels=tcn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        attention_reduction_ratio=attention_reduction_ratio
    )
    
    classifier_head = AttentionPoolingClassifierHead(
        input_dim=tcn_channels[-1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=classifier_dropout
    )
    
    return ModularMCATCNModel(feature_extractor, classifier_head, use_full_sequence=True)

def create_mcatcn_with_lstm_head(num_inputs, tcn_channels, num_classes, lstm_hidden_dim=None,
                               lstm_layers=1, kernel_size=2, dropout=0.2, attention_reduction_ratio=16, 
                               classifier_dropout=0.5):
    """
    Factory function to create MCA-TCN with LSTM head.
    """
    if lstm_hidden_dim is None:
        lstm_hidden_dim = tcn_channels[-1] // 2
    
    feature_extractor = MCATCNFeatureExtractor(
        num_inputs=num_inputs,
        num_channels=tcn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        attention_reduction_ratio=attention_reduction_ratio
    )
    
    classifier_head = LSTMClassifierHead(
        input_dim=tcn_channels[-1],
        hidden_dim=lstm_hidden_dim,
        num_classes=num_classes,
        num_layers=lstm_layers,
        dropout=classifier_dropout
    )
    
    return ModularMCATCNModel(feature_extractor, classifier_head, use_full_sequence=True)

def create_mcatcn_model_by_type(classifier_head_type, num_inputs, tcn_channels, num_classes,
                               kernel_size=2, dropout=0.2, attention_reduction_ratio=16, 
                               classifier_dropout=0.5):
    """
    Factory function to create MCA-TCN model with specified classifier head type.
    
    Args:
        classifier_head_type: Type of classifier head ('mlp', 'deep_mlp', 'attention_pooling', 'lstm')
        num_inputs: Number of input features
        tcn_channels: List of channel sizes for TCN layers
        num_classes: Number of output classes
        kernel_size: Kernel size for TCN
        dropout: Dropout rate for TCN
        attention_reduction_ratio: Reduction ratio for attention mechanism
        classifier_dropout: Dropout rate for classifier head
    
    Returns:
        ModularMCATCNModel instance with the specified head
    """
    if classifier_head_type == 'mlp':
        return create_mcatcn_with_mlp_head(
            num_inputs=num_inputs,
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            hidden_dim=tcn_channels[-1],
            kernel_size=kernel_size,
            dropout=dropout,
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        )
    elif classifier_head_type == 'deep_mlp':
        return create_mcatcn_with_deep_mlp_head(
            num_inputs=num_inputs,
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            hidden_dims=[tcn_channels[-1], tcn_channels[-1] // 2],
            kernel_size=kernel_size,
            dropout=dropout,
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        )
    elif classifier_head_type == 'attention_pooling':
        return create_mcatcn_with_attention_pooling_head(
            num_inputs=num_inputs,
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            hidden_dim=tcn_channels[-1] // 2,
            kernel_size=kernel_size,
            dropout=dropout,
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        )
    elif classifier_head_type == 'lstm':
        return create_mcatcn_with_lstm_head(
            num_inputs=num_inputs,
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            lstm_hidden_dim=tcn_channels[-1] // 2,
            lstm_layers=1,
            kernel_size=kernel_size,
            dropout=dropout,
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        )
    else:
        raise ValueError(f"Unsupported classifier_head_type: {classifier_head_type}. "
                        f"Supported types are: 'mlp', 'deep_mlp', 'attention_pooling', 'lstm'")

# --- Function to create MCA-TCN models with tuned classifier head parameters before data preparation function ---
def create_mcatcn_model_with_tuned_params(classifier_head_type, classifier_params, num_inputs, 
                                        tcn_channels, num_classes, kernel_size=2, dropout=0.2, 
                                        attention_reduction_ratio=16, classifier_dropout=0.5):
    """
    Create MCA-TCN model with tuned classifier head parameters.
    
    Args:
        classifier_head_type: Type of classifier head
        classifier_params: Dictionary of classifier-specific parameters from HPO
        num_inputs: Number of input features
        tcn_channels: List of channel sizes for TCN layers
        num_classes: Number of output classes
        kernel_size: Kernel size for TCN
        dropout: Dropout rate for TCN
        attention_reduction_ratio: Reduction ratio for attention mechanism
        classifier_dropout: Dropout rate for classifier head
    
    Returns:
        ModularMCATCNModel instance with tuned parameters
    """
    if classifier_head_type == 'mlp':
        hidden_dim = int(tcn_channels[-1] * classifier_params.get("hidden_multiplier", 1.0))
        return create_mcatcn_with_mlp_head(
            num_inputs=num_inputs,
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        )
    
    elif classifier_head_type == 'deep_mlp':
        # Create hidden dimensions with progressive reduction
        num_layers = classifier_params.get("num_layers", 2)
        dim_reduction = classifier_params.get("dim_reduction", 0.5)
        
        hidden_dims = []
        current_dim = tcn_channels[-1]
        for i in range(num_layers):
            hidden_dims.append(current_dim)
            current_dim = max(16, int(current_dim * dim_reduction))  # Minimum of 16 neurons
        
        return create_mcatcn_with_deep_mlp_head(
            num_inputs=num_inputs,
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout,
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        )
    
    elif classifier_head_type == 'attention_pooling':
        attention_hidden_dim = int(tcn_channels[-1] * classifier_params.get("attention_hidden_multiplier", 0.5))
        return create_mcatcn_with_attention_pooling_head(
            num_inputs=num_inputs,
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            hidden_dim=attention_hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        )
    
    elif classifier_head_type == 'lstm':
        lstm_hidden_dim = int(tcn_channels[-1] * classifier_params.get("lstm_hidden_multiplier", 0.5))
        lstm_num_layers = classifier_params.get("lstm_num_layers", 1)
        return create_mcatcn_with_lstm_head(
            num_inputs=num_inputs,
            tcn_channels=tcn_channels,
            num_classes=num_classes,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        )
    
    else:
        raise ValueError(f"Unsupported classifier_head_type: {classifier_head_type}")

# --- Dynamic Loss Weighting Function ---
def compute_dynamic_weighted_loss(criterion, outputs, targets, alpha=2.0, reduction='mean'):
    """
    Compute dynamic weighted loss based on confidence.
    Reduces the influence of high-loss samples by assigning smaller weights to them.
    
    Args:
        criterion: Loss criterion (e.g., nn.CrossEntropyLoss with reduction='none')
        outputs: Model outputs (logits)
        targets: Ground truth labels
        alpha: Alpha parameter for dynamic weighting (1-3 recommended range)
        reduction: How to reduce the final loss ('mean' or 'sum')
    
    Returns:
        Weighted loss tensor
    """
    # Get per-sample losses (no reduction)
    per_sample_losses = criterion(outputs, targets)
    
    # Compute confidence weights: smaller weight for high loss samples
    # weights = exp(-alpha * loss)
    weights = torch.exp(-alpha * per_sample_losses.detach())
    
    # Apply weights to losses
    weighted_losses = weights * per_sample_losses
    
    # Reduce according to specified reduction method
    if reduction == 'mean':
        return weighted_losses.mean()
    elif reduction == 'sum':
        return weighted_losses.sum()
    else:
        return weighted_losses

# --- Combined Loss Function with Self-Paced Learning ---
def compute_combined_weighted_loss(criterion, outputs, targets, batch_indices=None, 
                                 spl_learner=None, alpha=2.0, reduction='mean'):
    """
    Compute loss with both dynamic weighting and self-paced learning.
    
    Args:
        criterion: Loss criterion (e.g., nn.CrossEntropyLoss with reduction='none')
        outputs: Model outputs (logits)
        targets: Ground truth labels
        batch_indices: Indices of samples in this batch (for SPL)
        spl_learner: Self-Paced Learning tracker instance
        alpha: Alpha parameter for dynamic weighting
        reduction: How to reduce the final loss ('mean' or 'sum')
    
    Returns:
        Weighted loss tensor
    """
    # Get per-sample losses (no reduction)
    per_sample_losses = criterion(outputs, targets)
    
    # Start with equal weights
    weights = torch.ones_like(per_sample_losses)
    
    # Apply dynamic loss weighting if enabled
    if USE_DYNAMIC_LOSS_WEIGHTING:
        dynamic_weights = torch.exp(-alpha * per_sample_losses.detach())
        weights *= dynamic_weights
    
    # Apply self-paced learning weights if enabled
    if USE_SELF_PACED_LEARNING and spl_learner is not None and batch_indices is not None:
        spl_weights = spl_learner.get_sample_weights(batch_indices)
        weights *= spl_weights
    
    # Apply weights to losses
    weighted_losses = weights * per_sample_losses
    
    # Reduce according to specified reduction method
    if reduction == 'mean':
        return weighted_losses.mean()
    elif reduction == 'sum':
        return weighted_losses.sum()
    else:
        return weighted_losses

# --- Self-Paced Learning Class ---
class SelfPacedLearner:
    """
    Self-Paced Learning tracker that monitors sample classification history 
    and reduces weights for persistently misclassified samples over time.
    
    This helps avoid chasing potentially mislabeled or extremely difficult samples
    by gradually reducing their influence on training.
    """
    def __init__(self, num_samples, warmup_epochs=5, decay_rate=0.95, 
                 min_weight=0.1, memory_length=5, device='cpu'):
        """
        Initialize Self-Paced Learning tracker.
        
        Args:
            num_samples: Total number of training samples
            warmup_epochs: Number of epochs before starting history tracking
            decay_rate: Weight decay rate for persistently misclassified samples
            min_weight: Minimum weight for any sample (prevents complete exclusion)
            memory_length: Number of recent epochs to consider for history
            device: Device to store tensors on
        """
        self.num_samples = num_samples
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.memory_length = memory_length
        self.device = device
        
        # Track sample weights (initialized to 1.0)
        self.sample_weights = torch.ones(num_samples, device=device)
        
        # Track misclassification history (circular buffer)
        self.misclass_history = torch.zeros(num_samples, memory_length, device=device)
        
        # Track current epoch and position in circular buffer
        self.current_epoch = 0
        self.history_pos = 0
        
        # Track sample indices mapping from batches to global dataset
        self.sample_indices = None
        
        print(f"Initialized Self-Paced Learner for {num_samples} samples")
        print(f"  Warmup epochs: {warmup_epochs}")
        print(f"  Decay rate: {decay_rate}")
        print(f"  Min weight: {min_weight}")
        print(f"  Memory length: {memory_length} epochs")
    
    def set_sample_indices(self, indices):
        """Set the mapping from batch samples to global dataset indices."""
        self.sample_indices = torch.tensor(indices, device=self.device)
    
    def update_history(self, batch_indices, predictions, targets):
        """
        Update misclassification history for a batch of samples.
        
        Args:
            batch_indices: Indices of samples in this batch (relative to full dataset)
            predictions: Model predictions for this batch
            targets: True labels for this batch
        """
        if self.current_epoch < self.warmup_epochs:
            return  # Don't track during warmup
        
        # Convert to tensors if needed
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, device=self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device)
        if not isinstance(batch_indices, torch.Tensor):
            batch_indices = torch.tensor(batch_indices, device=self.device)
        
        # Calculate misclassifications (1 = misclassified, 0 = correct)
        misclassified = (predictions != targets).float()
        
        # Update history for these samples
        self.misclass_history[batch_indices, self.history_pos] = misclassified
    
    def update_weights(self):
        """
        Update sample weights based on misclassification history.
        Called at the end of each epoch.
        """
        if self.current_epoch < self.warmup_epochs:
            return
        
        # Calculate misclassification rate over recent epochs
        # Only consider epochs we've actually recorded
        epochs_recorded = min(self.current_epoch - self.warmup_epochs + 1, self.memory_length)
        if epochs_recorded == 0:
            return
        
        # Get relevant history (handle circular buffer)
        if epochs_recorded < self.memory_length:
            # We haven't filled the buffer yet
            relevant_history = self.misclass_history[:, :epochs_recorded]
        else:
            # Buffer is full, use all entries
            relevant_history = self.misclass_history
        
        # Calculate average misclassification rate per sample
        misclass_rate = relevant_history.mean(dim=1)  # Average over epochs
        
        # Update weights: reduce weight for samples with high misclassification rate
        # Apply decay proportional to misclassification rate
        weight_decay = self.decay_rate ** misclass_rate
        self.sample_weights *= weight_decay
        
        # Ensure weights don't go below minimum
        self.sample_weights = torch.clamp(self.sample_weights, min=self.min_weight)
    
    def advance_epoch(self):
        """Advance to next epoch and update circular buffer position."""
        self.current_epoch += 1
        if self.current_epoch >= self.warmup_epochs:
            self.history_pos = (self.history_pos + 1) % self.memory_length
    
    def get_sample_weights(self, batch_indices):
        """
        Get weights for a specific batch of samples.
        
        Args:
            batch_indices: Indices of samples in this batch
            
        Returns:
            Tensor of weights for the batch samples
        """
        if not isinstance(batch_indices, torch.Tensor):
            batch_indices = torch.tensor(batch_indices, device=self.device)
        
        return self.sample_weights[batch_indices]
    
    def reset(self):
        """Reset all tracking variables (useful for cross-validation folds)."""
        self.sample_weights.fill_(1.0)
        self.misclass_history.fill_(0.0)
        self.current_epoch = 0
        self.history_pos = 0
        self.sample_indices = None
    
    def get_statistics(self):
        """Get statistics about current sample weights and history."""
        stats = {
            'current_epoch': self.current_epoch,
            'mean_weight': self.sample_weights.mean().item(),
            'min_weight': self.sample_weights.min().item(),
            'max_weight': self.sample_weights.max().item(),
            'weights_below_threshold': (self.sample_weights < 0.5).sum().item(),
            'total_samples': self.num_samples
        }
        
        if self.current_epoch >= self.warmup_epochs:
            epochs_recorded = min(self.current_epoch - self.warmup_epochs + 1, self.memory_length)
            if epochs_recorded > 0:
                relevant_history = self.misclass_history[:, :epochs_recorded] if epochs_recorded < self.memory_length else self.misclass_history
                misclass_rate = relevant_history.mean(dim=1)
                stats.update({
                    'mean_misclass_rate': misclass_rate.mean().item(),
                    'max_misclass_rate': misclass_rate.max().item(),
                    'samples_high_misclass': (misclass_rate > 0.7).sum().item()
                })
        
        return stats
# --- Self-Paced Learning Class ---


# --- Data Preparation Function (MODIFIED for HPO) ---
def prepare_data_for_tcn_hpo(config, project_root):
    print("--- Starting Data Preparation for TCN (Harness for HPO) ---")

    intermediate_dir = os.path.join(project_root, config.get('intermediate_feature_dir', 'features'))
    x_windows_path = os.path.join(intermediate_dir, "X_windows_raw.npy")
    y_windows_path = os.path.join(intermediate_dir, "y_windows.npy")
    subject_ids_path = os.path.join(intermediate_dir, "subject_ids_windows.npy")

    required_files = [x_windows_path, y_windows_path, subject_ids_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required data file not found: {file_path}")
            return [None]*7 # Expected number of return values

    try:
        X_windows = np.load(x_windows_path, allow_pickle=True)
        y_windows = np.load(y_windows_path, allow_pickle=True)
        subject_ids_windows = np.load(subject_ids_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading data from .npy files: {e}")
        return [None]*7
    
    excluded_subjects_manual = config.get('excluded_subjects_manual', ['OutSense-036', 'OutSense-425', 'OutSense-515'])
    if excluded_subjects_manual:
        print(f"Attempting to manually exclude subjects: {excluded_subjects_manual}")
        initial_count = len(subject_ids_windows)
        exclusion_mask = ~np.isin(subject_ids_windows, excluded_subjects_manual)
        X_windows = X_windows[exclusion_mask]
        y_windows = y_windows[exclusion_mask]
        subject_ids_windows = subject_ids_windows[exclusion_mask]
        excluded_count = initial_count - len(subject_ids_windows)
        print(f"Manually excluded {excluded_count} data points belonging to specified subjects.")
        if X_windows.shape[0] == 0:
            print("ERROR: All data excluded after manual subject exclusion. Cannot proceed.")
            return [None]*7
        print(f"Data shape after manual exclusion: X={X_windows.shape}, y={y_windows.shape}, subjects={subject_ids_windows.shape[0] if subject_ids_windows.ndim > 0 else 0}")

    # --- START SENSOR EXCLUSION ---
    excluded_sensors = config.get('excluded_sensors', [])
    if excluded_sensors:
        print(f"Attempting to exclude sensors: {excluded_sensors}")
        
        # Get sensor column names from config if available
        sensor_columns = config.get('sensor_columns_original', [])
        
        if sensor_columns:
            print(f"Available sensor columns in config: {len(sensor_columns)} sensors")
            print(f"First few sensors: {sensor_columns[:10]}")
            
            # Find indices of sensors to exclude
            indices_to_exclude = []
            for excluded_sensor in excluded_sensors:
                if excluded_sensor in sensor_columns:
                    sensor_idx = sensor_columns.index(excluded_sensor)
                    indices_to_exclude.append(sensor_idx)
                    print(f"  Found sensor '{excluded_sensor}' at index {sensor_idx}")
                else:
                    print(f"  WARNING: Sensor '{excluded_sensor}' not found in sensor_columns_original")
            
            if indices_to_exclude:
                # Create a mask to keep all features except the excluded ones
                all_indices = list(range(X_windows.shape[2]))
                indices_to_keep = [idx for idx in all_indices if idx not in indices_to_exclude]
                
                print(f"Excluding {len(indices_to_exclude)} sensor features (indices: {indices_to_exclude})")
                print(f"Keeping {len(indices_to_keep)} sensor features out of {X_windows.shape[2]} total")
                
                # Apply sensor exclusion
                X_windows = X_windows[:, :, indices_to_keep]
                
                print(f"Data shape after sensor exclusion: X={X_windows.shape}")
                
                if X_windows.shape[2] == 0:
                    print("ERROR: All sensor features excluded. Cannot proceed.")
                    return [None]*7
            else:
                print("  No valid sensors found to exclude.")
        else:
            print("  WARNING: 'sensor_columns_original' not found in config. Cannot perform sensor exclusion by name.")
            print("  Sensor exclusion will be skipped. Consider adding sensor column names to config.yaml")
    # --- END SENSOR EXCLUSION ---

    # --- START LABEL REMAPPING ---
    print("INFO: Attempting to remap activity labels using Activity_Mapping_v2.csv")
    activity_mapping_path = os.path.join(project_root, "Activity_Mapping_v2.csv")
    if os.path.exists(activity_mapping_path):
        try:
            mapping_df = pd.read_csv(activity_mapping_path)
            if 'Former_Label' in mapping_df.columns and 'New_Label' in mapping_df.columns:
                label_mapping_dict = pd.Series(mapping_df.New_Label.values, index=mapping_df.Former_Label).to_dict()
                y_series = pd.Series(y_windows)
                y_windows_mapped = y_series.map(label_mapping_dict).fillna(y_series).values
                changed_count = np.sum(y_windows != y_windows_mapped)
                print(f"  Successfully remapped {changed_count} labels out of {len(y_windows)} based on Activity_Mapping_v2.csv.")
                
                if changed_count > 0:
                    diff_indices = np.where(y_windows != y_windows_mapped)[0]
                    print(f"  Example of remapping (first {min(5, len(diff_indices))} changes):")
                    for i in range(min(5, len(diff_indices))):
                        idx = diff_indices[i]
                        print(f"    Original: '{y_windows[idx]}' -> Mapped: '{y_windows_mapped[idx]}'")
                
                y_windows = y_windows_mapped
            else:
                print("  Warning: Activity_Mapping_v2.csv does not contain 'Former_Label' and/or 'New_Label' columns. Skipping remapping.")
        except Exception as e:
            print(f"  Error processing Activity_Mapping_v2.csv: {e}. Skipping remapping.")
    else:
        print(f"  Warning: Activity_Mapping_v2.csv not found at {activity_mapping_path}. Skipping remapping.")
    # --- END LABEL REMAPPING ---

    if not isinstance(X_windows, np.ndarray) or not isinstance(y_windows, np.ndarray) or not isinstance(subject_ids_windows, np.ndarray):
        print("Error: Loaded data is not in the expected NumPy array format.")
        return [None]*7

    print("INFO: Unconditionally removing 'Other' and 'Unknown' classes.")
    other_class_label = "Other"
    if other_class_label in np.unique(y_windows):
        initial_count_other = len(y_windows)
        other_mask = y_windows != other_class_label
        X_windows = X_windows[other_mask]; subject_ids_windows = subject_ids_windows[other_mask]; y_windows = y_windows[other_mask]
        print(f"  Removed {initial_count_other - len(y_windows)} windows for '{other_class_label}'.")
        if len(y_windows) == 0: print("ERROR: All data removed after 'Other' class exclusion."); return [None]*7
    
    unknown_class_label = "Unknown"
    if unknown_class_label in np.unique(y_windows):
        initial_count_unknown = len(y_windows)
        unknown_mask = y_windows != unknown_class_label
        X_windows = X_windows[unknown_mask]; subject_ids_windows = subject_ids_windows[unknown_mask]; y_windows = y_windows[unknown_mask]
        print(f"  Removed {initial_count_unknown - len(y_windows)} windows for '{unknown_class_label}'.")
        if len(y_windows) == 0: print("ERROR: All data removed after 'Unknown' class exclusion."); return [None]*7

    selected_classes = config.get('selected_classes', ['Propulsion', 'Resting', 'Transfer', 'Exercising', 'Conversation'])
    if selected_classes:
        print(f"INFO: Filtering dataset to include only: {selected_classes}")
        initial_count_y = len(y_windows)
        class_selection_mask = np.isin(y_windows, selected_classes)
        X_windows = X_windows[class_selection_mask]; y_windows = y_windows[class_selection_mask]; subject_ids_windows = subject_ids_windows[class_selection_mask]
        print(f"Filtered by 'selected_classes': Kept {len(y_windows)} from {initial_count_y} windows.")
        if len(y_windows) == 0: print("ERROR: All data removed after 'selected_classes' filtering."); return [None]*7
    
    min_instances_threshold = config.get('min_class_instances', 10)
    if min_instances_threshold > 0 and len(y_windows) > 0:
        print(f"INFO: Removing classes with fewer than {min_instances_threshold} instances.")
        unique_labels, counts = np.unique(y_windows, return_counts=True)
        labels_to_keep = unique_labels[counts >= min_instances_threshold]
        if len(labels_to_keep) < len(unique_labels):
            instance_mask = np.isin(y_windows, labels_to_keep)
            X_windows = X_windows[instance_mask]; y_windows = y_windows[instance_mask]; subject_ids_windows = subject_ids_windows[instance_mask]
            print(f"  Removed {len(unique_labels) - len(labels_to_keep)} classes. Kept {len(labels_to_keep)} classes.")
            if len(y_windows) == 0: print("ERROR: All data removed after min_class_instances filtering."); return [None]*7
        else:
            print("  All classes meet the minimum instance threshold.")

    if X_windows.ndim != 3:
        print(f"ERROR: X_windows has incorrect dimensions {X_windows.ndim}, expected 3 (samples, sequence_length, features).")
        return [None]*7
    n_features_loaded = X_windows.shape[2]
    print(f"Loaded {X_windows.shape[0]} windows, seq_len {X_windows.shape[1]}, {n_features_loaded} features after filtering.")

    if X_windows.dtype != np.float32:
        print(f"INFO: Converting X_windows from {X_windows.dtype} to np.float32.")
        X_windows = X_windows.astype(np.float32)

    test_subjects = config.get('test_subjects', [])
    if not test_subjects: 
        print("ERROR: 'test_subjects' not defined in config. Cannot proceed.")
        return [None]*7
    
    test_mask = np.isin(subject_ids_windows, test_subjects)
    train_val_mask = ~test_mask
    
    X_train_val_all = X_windows[train_val_mask]
    y_train_val_all = y_windows[train_val_mask]
    subject_ids_train_val_all = subject_ids_windows[train_val_mask]
    
    X_test = X_windows[test_mask]
    y_test = y_windows[test_mask]
    
    if len(X_train_val_all) == 0:
        print("ERROR: No data remains for training/validation after test set separation.")
    if len(X_test) == 0:
        print("WARNING: No data for the test set based on provided test_subjects.")

    print(f"Data split for HPO: Train/Val Pool={len(X_train_val_all)}, Test={len(X_test)} sequences.")

    # Determine all unique labels present in the data that will be used by an encoder
    combined_y_for_labels = []
    if len(y_train_val_all) > 0: combined_y_for_labels.append(y_train_val_all)
    if len(y_test) > 0: combined_y_for_labels.append(y_test)

    if not combined_y_for_labels:
        print("ERROR: No labels found in either training/validation pool or test set after filtering.")
        return [None]*7
        
    all_potential_labels = np.unique(np.concatenate(combined_y_for_labels))
    
    if len(all_potential_labels) == 0:
        print("ERROR: No unique labels found after combining train/val and test sets.")
        return [None]*7
        
    print(f"Found {len(all_potential_labels)} unique labels across combined data for encoder fitting: {all_potential_labels[:10]}...")

    print("Data preparation for HPO harness complete.")
    return X_train_val_all, y_train_val_all, subject_ids_train_val_all, X_test, y_test, n_features_loaded, all_potential_labels



# --- Optuna Objective Function ---
def objective(trial, X_tv_data, y_tv_data, groups_tv_data, n_features, potential_labels_for_encoder_fit, device, min_class_instances_config):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    tcn_kernel_size = trial.suggest_categorical("tcn_kernel_size", [2, 3, 5])
    
    # MCA-TCN CHANGE: Add attention reduction ratio as a tunable hyperparameter
    attention_reduction_ratio = trial.suggest_categorical("attention_reduction_ratio", [4, 8, 16])
    
    # Dynamic Loss Weighting hyperparameter
    dynamic_loss_alpha = trial.suggest_float("dynamic_loss_alpha", 1.0, 3.0) if USE_DYNAMIC_LOSS_WEIGHTING else DYNAMIC_LOSS_ALPHA
    
    # Self-Paced Learning hyperparameters
    if USE_SELF_PACED_LEARNING:
        spl_decay_rate = trial.suggest_float("spl_decay_rate", 0.90, 0.99)
        spl_min_weight = trial.suggest_float("spl_min_weight", 0.05, 0.3)
    else:
        spl_decay_rate = SPL_DECAY_RATE
        spl_min_weight = SPL_MIN_WEIGHT
    
    # NEW: Classifier Head Type Selection and Hyperparameters
    classifier_head_type = trial.suggest_categorical("classifier_head_type", ["mlp", "deep_mlp"])
    classifier_dropout = trial.suggest_float("classifier_dropout", 0.1, 0.7)

    # Classifier-specific hyperparameters
    classifier_params = {}
    if classifier_head_type == "mlp":
        # For simple MLP, tune the hidden dimension
        mlp_hidden_multiplier = trial.suggest_float("mlp_hidden_multiplier", 0.5, 2.0)
        classifier_params["hidden_multiplier"] = mlp_hidden_multiplier
        
    elif classifier_head_type == "deep_mlp":
        # For deep MLP, tune the number of layers and dimension reduction
        num_hidden_layers = trial.suggest_int("deep_mlp_num_layers", 2, 6)
        hidden_dim_reduction = trial.suggest_float("deep_mlp_dim_reduction", 0.3, 0.8)
        classifier_params["num_layers"] = num_hidden_layers
        classifier_params["dim_reduction"] = hidden_dim_reduction
        
    elif classifier_head_type == "attention_pooling":
        # For attention pooling, tune the attention hidden dimension
        attention_hidden_multiplier = trial.suggest_float("attention_hidden_multiplier", 0.25, 1.0)
        classifier_params["attention_hidden_multiplier"] = attention_hidden_multiplier
        
    elif classifier_head_type == "lstm":
        # For LSTM head, tune hidden dimension and number of layers
        lstm_hidden_multiplier = trial.suggest_float("lstm_hidden_multiplier", 0.25, 1.0)
        lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 3)
        classifier_params["lstm_hidden_multiplier"] = lstm_hidden_multiplier
        classifier_params["lstm_num_layers"] = lstm_num_layers
    
    # Tune number of TCN layers (2-6 layers)
    num_tcn_layers = trial.suggest_int("num_tcn_layers", 2, 6)
    
    # Define channels for maximum possible layers (6), then use only the first num_tcn_layers
    max_layers = 6
    all_channels = []
    for layer_idx in range(max_layers):
        if layer_idx == 0:
            # First layer - smaller channels
            channels = trial.suggest_categorical(f"num_channels_l{layer_idx+1}", [16, 32, 64])
        elif layer_idx < max_layers - 1:
            # Middle layers - medium channels
            channels = trial.suggest_categorical(f"num_channels_l{layer_idx+1}", [32, 64, 128])
        else:
            # Last layer - larger channels
            channels = trial.suggest_categorical(f"num_channels_l{layer_idx+1}", [64, 128, 256])
        all_channels.append(channels)
    
    # Use only the first num_tcn_layers channels
    tcn_num_channels = all_channels[:num_tcn_layers]
    
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    
    # Learning rate scheduler hyperparameters
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["ReduceLROnPlateau", "CosineAnnealingLR"]) if use_scheduler else None
    
    # Scheduler-specific parameters
    scheduler_params = {}
    if use_scheduler:
        if scheduler_type == "ReduceLROnPlateau":
            scheduler_params["factor"] = trial.suggest_float("scheduler_factor", 0.1, 0.8)
            scheduler_params["patience"] = trial.suggest_int("scheduler_patience", 3, 8)
            scheduler_params["min_lr"] = trial.suggest_float("scheduler_min_lr", 1e-7, 1e-5, log=True)
        elif scheduler_type == "CosineAnnealingLR":
            scheduler_params["T_max"] = trial.suggest_int("scheduler_T_max", 10, HPO_MAX_EPOCHS)
            scheduler_params["eta_min"] = trial.suggest_float("scheduler_eta_min", 1e-7, 1e-5, log=True)

    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tv_data, y_tv_data, groups_tv_data)):
        print(f"--- HPO Trial {trial.number}, Fold {fold+1}/{CV_N_SPLITS} ---")
        X_train_fold, X_val_fold = X_tv_data[train_idx], X_tv_data[val_idx]
        y_train_fold, y_val_fold = y_tv_data[train_idx], y_tv_data[val_idx]
        groups_train_fold = groups_tv_data[train_idx]

        # --- START UNDERSAMPLING FOR THIS FOLD'S TRAINING DATA ---
        if len(X_train_fold) > 0:
            unique_classes_fold, counts_fold = np.unique(y_train_fold, return_counts=True)
            print(f"    Fold {fold+1} train class distribution before undersampling: {dict(zip(unique_classes_fold, counts_fold))}")

            # Filter out classes below min_class_instances_config for undersampling
            if min_class_instances_config > 0 and len(unique_classes_fold) > 0:
                labels_to_remove_mask = counts_fold < min_class_instances_config
                labels_to_remove = unique_classes_fold[labels_to_remove_mask]
                if len(labels_to_remove) > 0:
                    print(f"      Removing classes from fold train set (for undersampling) with < {min_class_instances_config} instances: {list(labels_to_remove)}")
                    train_keep_mask = ~np.isin(y_train_fold, labels_to_remove)
                    X_train_fold = X_train_fold[train_keep_mask]
                    y_train_fold = y_train_fold[train_keep_mask]
                    groups_train_fold = groups_train_fold[train_keep_mask]
                    if len(y_train_fold) == 0:
                        print("      Warning: Fold training set empty after removing low-instance classes. Skipping fold.")
                        continue # Skip to next fold
                    unique_classes_fold, counts_fold = np.unique(y_train_fold, return_counts=True)
                    print(f"      Fold {fold+1} train class distribution after filtering for >= {min_class_instances_config} instances: {dict(zip(unique_classes_fold, counts_fold))}")

            # Proceed with undersampling if multiple classes remain
            if len(unique_classes_fold) > 1:
                min_class_count_fold = np.min(counts_fold)
                print(f"    Minority class count for undersampling in fold {fold+1}: {min_class_count_fold}")
                
                resampled_X_fold_list, resampled_y_fold_list, resampled_groups_fold_list = [], [], []
                for cls_label in unique_classes_fold:
                    cls_indices = np.where(y_train_fold == cls_label)[0]
                    selected_indices = np.random.choice(cls_indices, size=min_class_count_fold, replace=False)
                    resampled_X_fold_list.append(X_train_fold[selected_indices])
                    resampled_y_fold_list.append(y_train_fold[selected_indices])
                    resampled_groups_fold_list.append(groups_train_fold[selected_indices])
                
                if resampled_X_fold_list:
                    X_train_fold = np.concatenate(resampled_X_fold_list, axis=0)
                    y_train_fold = np.concatenate(resampled_y_fold_list, axis=0)
                    groups_train_fold = np.concatenate(resampled_groups_fold_list, axis=0)

                    shuffle_indices = np.random.permutation(len(X_train_fold))
                    X_train_fold = X_train_fold[shuffle_indices]
                    y_train_fold = y_train_fold[shuffle_indices]
                    groups_train_fold = groups_train_fold[shuffle_indices]
                    print(f"    Resampled fold {fold+1} training set size: {len(X_train_fold)}")
                    if len(X_train_fold) == 0:
                        print("      Warning: Fold training set empty after undersampling. Skipping fold.")
                        continue # Skip to next fold
                    unique_classes_resampled_f, counts_resampled_f = np.unique(y_train_fold, return_counts=True)
                    print(f"    Resampled fold {fold+1} train class distribution: {dict(zip(unique_classes_resampled_f, counts_resampled_f))}")
            elif len(unique_classes_fold) == 1:
                print(f"    Skipping undersampling for fold {fold+1} as only one class remains after filtering.")
            else:
                print(f"    Skipping undersampling for fold {fold+1} as no classes remain after filtering.")
                if len(X_train_fold) == 0: # Double check if it became empty
                    print("      Warning: Fold training set is empty before model training. Skipping fold.")
                    continue # Skip to next fold
        else: # X_train_fold was initially empty
            print(f"    Skipping undersampling for fold {fold+1} as training set is initially empty.")
            continue # Skip to next fold
        # --- END UNDERSAMPLING FOR THIS FOLD'S TRAINING DATA ---

        # Scaler and LabelEncoder for the fold
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold.reshape(-1, n_features)).reshape(X_train_fold.shape)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold.reshape(-1, n_features)).reshape(X_val_fold.shape)

        label_encoder_fold = LabelEncoder()
        label_encoder_fold.fit(potential_labels_for_encoder_fit)
        y_train_fold_enc = label_encoder_fold.transform(y_train_fold)
        y_val_fold_enc = label_encoder_fold.transform(y_val_fold)
        num_classes_fold = len(label_encoder_fold.classes_)

        X_train_tensor = torch.from_numpy(X_train_fold_scaled.transpose(0, 2, 1)).float()
        y_train_tensor = torch.from_numpy(y_train_fold_enc).long()
        train_loader_fold = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

        X_val_tensor = torch.from_numpy(X_val_fold_scaled.transpose(0, 2, 1)).float()
        y_val_tensor = torch.from_numpy(y_val_fold_enc).long()
        val_loader_fold = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

        # MCA-TCN CHANGE: Instantiate the modular MCATCNModel with tuned hyperparameters
        model_fold = create_mcatcn_model_with_tuned_params(
            classifier_head_type=classifier_head_type,
            classifier_params=classifier_params,
            num_inputs=n_features, 
            tcn_channels=tcn_num_channels, 
            num_classes=num_classes_fold, 
            kernel_size=tcn_kernel_size, 
            dropout=dropout_rate, 
            attention_reduction_ratio=attention_reduction_ratio,
            classifier_dropout=classifier_dropout
        ).to(device)
        optimizer_fold = optim.AdamW(model_fold.parameters(), lr=lr, weight_decay=weight_decay)
        # Use reduction='none' for dynamic loss weighting or SPL, otherwise use default
        criterion_fold = nn.CrossEntropyLoss(reduction='none' if (USE_DYNAMIC_LOSS_WEIGHTING or USE_SELF_PACED_LEARNING) else 'mean')

        # Initialize scheduler if enabled
        scheduler_fold = None
        if use_scheduler:
            if scheduler_type == "ReduceLROnPlateau":
                scheduler_fold = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_fold, 
                    mode='min', 
                    factor=scheduler_params["factor"],
                    patience=scheduler_params["patience"],
                    min_lr=scheduler_params["min_lr"],
                    verbose=False
                )
            elif scheduler_type == "CosineAnnealingLR":
                scheduler_fold = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_fold,
                    T_max=scheduler_params["T_max"],
                    eta_min=scheduler_params["eta_min"]
                )

        best_val_loss_fold = float('inf')
        epochs_no_improve_fold = 0
        
        # Initialize Self-Paced Learning tracker if enabled
        spl_learner = None
        if USE_SELF_PACED_LEARNING:
            spl_learner = SelfPacedLearner(
                num_samples=len(X_train_fold),
                warmup_epochs=SPL_WARMUP_EPOCHS,
                decay_rate=spl_decay_rate,
                min_weight=spl_min_weight,
                memory_length=SPL_MEMORY_LENGTH,
                device=device
            )
        
        for epoch in range(HPO_MAX_EPOCHS):
            model_fold.train()
            train_loss_epoch = 0
            
            # Track batch indices for SPL
            batch_start_idx = 0
            
            for batch_X, batch_y in train_loader_fold:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_size = batch_X.size(0)
                batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
                
                optimizer_fold.zero_grad()
                outputs = model_fold(batch_X)
                
                # Apply combined loss weighting (dynamic + SPL)
                if USE_DYNAMIC_LOSS_WEIGHTING or USE_SELF_PACED_LEARNING:
                    loss = compute_combined_weighted_loss(
                        criterion_fold, outputs, batch_y, 
                        batch_indices=batch_indices,
                        spl_learner=spl_learner,
                        alpha=dynamic_loss_alpha
                    )
                else:
                    loss = criterion_fold(outputs, batch_y)
                
                # Update SPL history if enabled
                if USE_SELF_PACED_LEARNING and spl_learner is not None:
                    with torch.no_grad():
                        predictions = torch.argmax(outputs, dim=1)
                        spl_learner.update_history(batch_indices, predictions, batch_y)
                
                loss.backward()
                optimizer_fold.step()
                train_loss_epoch += loss.item()
                
                batch_start_idx += batch_size
            
            # Update SPL weights at end of epoch
            if USE_SELF_PACED_LEARNING and spl_learner is not None:
                spl_learner.update_weights()
                spl_learner.advance_epoch()
            
            model_fold.eval(); val_loss_epoch = 0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader_fold:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model_fold(batch_X_val)
                    
                    # For validation, use standard loss (not dynamic weighting)
                    if USE_DYNAMIC_LOSS_WEIGHTING:
                        # Use mean reduction for validation loss
                        loss_val = criterion_fold(outputs_val, batch_y_val).mean()
                    else:
                        loss_val = criterion_fold(outputs_val, batch_y_val)
                    
                    val_loss_epoch += loss_val.item()
            avg_val_loss = val_loss_epoch / len(val_loader_fold) if len(val_loader_fold) > 0 else float('inf')
            
            # Step scheduler
            if scheduler_fold is not None:
                if scheduler_type == "ReduceLROnPlateau":
                    scheduler_fold.step(avg_val_loss)
                elif scheduler_type == "CosineAnnealingLR":
                    scheduler_fold.step()
            
            if avg_val_loss < best_val_loss_fold - EARLY_STOPPING_MIN_DELTA:
                best_val_loss_fold = avg_val_loss; epochs_no_improve_fold = 0
            else:
                epochs_no_improve_fold += 1
            if epochs_no_improve_fold >= HPO_PATIENCE: print(f"Early stopping E{epoch+1} T{trial.number} F{fold+1}."); break
        
        model_fold.eval(); all_preds_fold, all_labels_fold = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader_fold:
                outputs = model_fold(batch_X.to(device)); preds = torch.argmax(outputs, dim=1)
                all_preds_fold.extend(preds.cpu().numpy()); all_labels_fold.extend(batch_y.cpu().numpy())
        
        f1_val_fold = f1_score(all_labels_fold, all_preds_fold, average='weighted', zero_division=0) if len(all_labels_fold) > 0 else 0.0
        fold_f1_scores.append(f1_val_fold)
        print(f"T{trial.number} F{fold+1} Val F1: {f1_val_fold:.4f}")
        trial.report(f1_val_fold, fold)
        if trial.should_prune(): print(f"T{trial.number} pruned F{fold+1}."); raise optuna.TrialPruned()

    avg_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0.0
    print(f"T{trial.number} Avg Val F1: {avg_f1:.4f}")
    return avg_f1

# --- Training Function (from original, adapted for early stopping) ---
def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, device, num_epochs, best_model_save_path, last_model_save_path, scheduler=None, dynamic_loss_alpha=None, spl_decay_rate=None, spl_min_weight=None):
    print("\\n--- Starting Model Training with Early Stopping ---")
    
    if USE_DYNAMIC_LOSS_WEIGHTING:
        alpha_value = dynamic_loss_alpha if dynamic_loss_alpha is not None else DYNAMIC_LOSS_ALPHA
        print(f"Using Dynamic Loss Weighting with alpha = {alpha_value:.3f}")
        print("This reduces the influence of high-loss samples to focus on more trustworthy samples.")
    else:
        print("Using standard Cross-Entropy Loss (no dynamic weighting)")
    
    if USE_SELF_PACED_LEARNING:
        decay_rate = spl_decay_rate if spl_decay_rate is not None else SPL_DECAY_RATE
        min_weight = spl_min_weight if spl_min_weight is not None else SPL_MIN_WEIGHT
        print(f"Using Self-Paced Learning with decay_rate = {decay_rate:.3f}, min_weight = {min_weight:.3f}")
        print("This reduces weights for persistently misclassified samples over time.")
    
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_dict = None
    
    # Initialize Self-Paced Learning tracker if enabled
    spl_learner = None
    if USE_SELF_PACED_LEARNING:
        # Estimate number of training samples
        num_train_samples = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else len(train_loader) * train_loader.batch_size
        decay_rate = spl_decay_rate if spl_decay_rate is not None else SPL_DECAY_RATE
        min_weight = spl_min_weight if spl_min_weight is not None else SPL_MIN_WEIGHT
        
        spl_learner = SelfPacedLearner(
            num_samples=num_train_samples,
            warmup_epochs=SPL_WARMUP_EPOCHS,
            decay_rate=decay_rate,
            min_weight=min_weight,
            memory_length=SPL_MEMORY_LENGTH,
            device=device
        )
    
    # Track loss histories for all three sets
    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        # Track batch indices for SPL
        batch_start_idx = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_size = batch_X.size(0)
            batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Apply combined loss weighting (dynamic + SPL)
            if USE_DYNAMIC_LOSS_WEIGHTING or USE_SELF_PACED_LEARNING:
                alpha = dynamic_loss_alpha if dynamic_loss_alpha is not None else DYNAMIC_LOSS_ALPHA
                loss = compute_combined_weighted_loss(
                    criterion, outputs, batch_y, 
                    batch_indices=batch_indices,
                    spl_learner=spl_learner,
                    alpha=alpha
                )
            else:
                loss = criterion(outputs, batch_y)
            
            # Update SPL history if enabled
            if USE_SELF_PACED_LEARNING and spl_learner is not None:
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1)
                    spl_learner.update_history(batch_indices, predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
            batch_start_idx += batch_size
        
        # Update SPL weights at end of epoch
        if USE_SELF_PACED_LEARNING and spl_learner is not None:
            spl_learner.update_weights()
            spl_learner.advance_epoch()
            
            # Print SPL statistics periodically
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                stats = spl_learner.get_statistics()
                print(f"  SPL Stats - Epoch {epoch+1}: Mean weight: {stats['mean_weight']:.3f}, "
                      f"Low weights: {stats['weights_below_threshold']}/{stats['total_samples']}")
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        if val_loader: # Only validate if val_loader is provided
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model(batch_X_val)
                    
                    # For validation, use standard loss (not dynamic weighting)
                    if USE_DYNAMIC_LOSS_WEIGHTING:
                        # Use mean reduction for validation loss
                        loss_val = criterion(outputs_val, batch_y_val).mean()
                    else:
                        loss_val = criterion(outputs_val, batch_y_val)
                    
                    epoch_val_loss += loss_val.item()
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Test loss computation (for monitoring only, doesn't affect early stopping)
            epoch_test_loss = 0
            if test_loader and DISPLAY_TEST_LOSS_DURING_TRAINING:
                with torch.no_grad():
                    for batch_X_test, batch_y_test in test_loader:
                        batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                        outputs_test = model(batch_X_test)
                        
                        # For test monitoring, use standard loss (not dynamic weighting)
                        if USE_DYNAMIC_LOSS_WEIGHTING:
                            # Use mean reduction for test loss
                            loss_test = criterion(outputs_test, batch_y_test).mean()
                        else:
                            loss_test = criterion(outputs_test, batch_y_test)
                        
                        epoch_test_loss += loss_test.item()
                avg_test_loss = epoch_test_loss / len(test_loader)
                test_losses.append(avg_test_loss)
                
                # Print progress with all three losses
                if epoch % PRINT_EPOCH_FREQUENCY == 0 or epoch == num_epochs - 1:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
            else:
                test_losses.append(None)  # Placeholder when test loss is not computed
                if epoch % PRINT_EPOCH_FREQUENCY == 0 or epoch == num_epochs - 1:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Step scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:  # CosineAnnealingLR or other schedulers
                    scheduler.step()

            if avg_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state_dict = copy.deepcopy(model.state_dict())
                torch.save(best_model_state_dict, best_model_save_path)
                print(f"Validation loss improved. Saved best model to {best_model_save_path}")
            else:
                epochs_no_improve += 1
        else: # No validation loader, just train and log
            val_losses.append(None)
            test_losses.append(None)
            if epoch % PRINT_EPOCH_FREQUENCY == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} (No validation)")
            # Step scheduler even without validation for schedulers that don't need validation loss
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        if val_loader and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    print(f"Saving last model state to {last_model_save_path}.")
    torch.save(model.state_dict(), last_model_save_path)

    if best_model_state_dict:
        print("Loading best model weights for returning.")
        model.load_state_dict(best_model_state_dict)
    else: # Should only happen if no val_loader or no improvement ever
        print("No best model state recorded (or no validation), returning model in its last state.")
        if os.path.exists(best_model_save_path) and val_loader: # If best was saved but loop ended due to num_epochs
             model.load_state_dict(torch.load(best_model_save_path)) # Ensure best is loaded

    return model, last_model_save_path, train_losses, val_losses, test_losses

# --- Evaluation Functions (from original) ---
def evaluate_model(model, test_loader, label_encoder, device, filename_suffix=""):
    # MCA-TCN CHANGE: Updated filenames for clarity
    print(f"\\n--- Evaluating Model (Standard) {filename_suffix} ---")
    model.eval() 
    all_preds_encoded, all_labels_encoded = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds_encoded.extend(preds.cpu().numpy())
            all_labels_encoded.extend(batch_y.cpu().numpy())

    if not all_labels_encoded:
        print("No labels in test set for evaluation. Skipping.")
        return

    all_preds_encoded = np.array(all_preds_encoded)
    all_labels_encoded = np.array(all_labels_encoded)
    
    print(f"\\n--- Overall Performance {filename_suffix} ---")
    accuracy_all = accuracy_score(all_labels_encoded, all_preds_encoded)
    f1_all = f1_score(all_labels_encoded, all_preds_encoded, average='weighted', zero_division=0)
    print(f"Test Accuracy: {accuracy_all:.4f}")
    print(f"Test F1-Score (Weighted): {f1_all:.4f}")
    
    # Step 1: Detailed Per-Class Performance Metrics
    print("\n--- Per-Class Performance ---")
    # Use the label_encoder to get the actual class names
    class_names = label_encoder.classes_ 
    report = classification_report(all_labels_encoded, all_preds_encoded, target_names=class_names, zero_division=0)
    print(report)

    # Save the classification report to a file
    with open(f"mcatcn_classification_report{filename_suffix}.txt", "w") as f:
        f.write(report)
    print(f"Classification report saved to mcatcn_classification_report{filename_suffix}.txt")
    
    cm_all = confusion_matrix(all_labels_encoded, all_preds_encoded, labels=np.arange(len(label_encoder.classes_)))
    
    plt.figure(figsize=(25, 22))
    all_class_names = label_encoder.classes_
    num_all_classes = len(all_class_names)
    annot_all = num_all_classes <= 30
    tick_step = 1 if num_all_classes <= 30 else max(1, num_all_classes // 30)
    xticklabels_all = all_class_names[::tick_step] if num_all_classes > 30 else all_class_names
    yticklabels_all = all_class_names[::tick_step] if num_all_classes > 30 else all_class_names
    
    sns.heatmap(cm_all, annot=annot_all, cmap='Blues', 
                xticklabels=xticklabels_all, yticklabels=yticklabels_all, fmt='d' if annot_all else '')
    plt.title(f'Confusion Matrix{filename_suffix}'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    cm_all_filename = f"mcatcn_confusion_matrix{filename_suffix}.png" # MCA-TCN CHANGE
    plt.savefig(cm_all_filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Confusion matrix saved to: {cm_all_filename}")

    cm_all_normalized = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]
    cm_all_normalized = np.nan_to_num(cm_all_normalized, nan=0.0)
    plt.figure(figsize=(25, 22))
    sns.heatmap(cm_all_normalized, annot=annot_all, cmap='Blues', 
                xticklabels=xticklabels_all, yticklabels=yticklabels_all, fmt='.2f' if annot_all else '')
    plt.title(f'Normalized Confusion Matrix{filename_suffix}'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    cm_all_normalized_filename = f"mcatcn_normalized_confusion_matrix{filename_suffix}.png" # MCA-TCN CHANGE
    plt.savefig(cm_all_normalized_filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Normalized confusion matrix saved to: {cm_all_normalized_filename}")

def evaluate_with_uncertainty(model, test_loader, label_encoder, device, filename_suffix=""):
    if MC_DROPOUT_SAMPLES <= 0:
        print("MC_DROPOUT_SAMPLES is 0 or less. Skipping uncertainty evaluation.")
        return
        
    print(f"\\n--- Evaluating Model with Uncertainty (MC Dropout {MC_DROPOUT_SAMPLES} samples) {filename_suffix} ---")
    model.enable_dropout()
    all_labels_encoded = []
    all_mc_softmax_probs_list = [] # List to store softmax probs from each batch

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_mc_softmax_probs = [] # (n_mc_samples, batch_size, num_classes)
            for _ in range(MC_DROPOUT_SAMPLES):
                outputs = model(batch_X)
                softmax_probs = torch.softmax(outputs, dim=1)
                batch_mc_softmax_probs.append(softmax_probs.cpu().numpy())
            
            all_mc_softmax_probs_list.append(np.array(batch_mc_softmax_probs))
            all_labels_encoded.extend(batch_y.cpu().numpy()) # batch_y is already on CPU from DataLoader or moved

    if not all_labels_encoded:
        print("No labels in test set for uncertainty evaluation. Skipping.")
        model.eval() # Set back to eval mode
        return

    # Concatenate along the batch dimension (axis=1 because shape is (n_mc_samples, batch_size, num_classes))
    all_mc_probabilities = np.concatenate(all_mc_softmax_probs_list, axis=1) 
    # Transpose to (total_samples, n_mc_samples, num_classes)
    all_mc_probabilities = all_mc_probabilities.transpose(1, 0, 2)
    all_labels_encoded = np.array(all_labels_encoded)

    mean_probs_per_sample = all_mc_probabilities.mean(axis=1) 
    final_preds_encoded = np.argmax(mean_probs_per_sample, axis=1)
    predictive_entropy = -np.sum(mean_probs_per_sample * np.log(mean_probs_per_sample + 1e-9), axis=1)
    
    uncertainty_df = pd.DataFrame({
        'true_label': label_encoder.inverse_transform(all_labels_encoded),
        'predicted_label': label_encoder.inverse_transform(final_preds_encoded),
        'is_correct': all_labels_encoded == final_preds_encoded,
        'predictive_entropy': predictive_entropy
    })

    print("\\n--- Uncertainty Analysis Summary ---")
    if not uncertainty_df.empty:
        print(uncertainty_df.groupby('true_label')['predictive_entropy'].describe())
        
        # Step 2: Granular Uncertainty Analysis
        print("\n--- Granular Uncertainty Analysis ---")
        # 1. Average uncertainty per true class
        avg_uncertainty_by_class = uncertainty_df.groupby('true_label')['predictive_entropy'].mean().sort_values(ascending=False)
        print("\nAverage Predictive Entropy by True Class (Higher is more uncertain):")
        print(avg_uncertainty_by_class)

        # 2. Average uncertainty for correct vs. incorrect predictions per class
        uncertainty_by_correctness = uncertainty_df.groupby(['true_label', 'is_correct'])['predictive_entropy'].mean().unstack()
        print("\nAverage Predictive Entropy (Correct vs. Incorrect):")
        print(uncertainty_by_correctness)

        # Save these results to CSV for easier analysis
        avg_uncertainty_by_class.to_csv(f"mcatcn_avg_uncertainty_by_class{filename_suffix}.csv")
        uncertainty_by_correctness.to_csv(f"mcatcn_uncertainty_by_correctness{filename_suffix}.csv")
        print(f"Uncertainty analysis data saved to CSV files with suffix: {filename_suffix}")
    else:
        print("Uncertainty DataFrame is empty.")

    if not uncertainty_df.empty and 'predictive_entropy' in uncertainty_df.columns and 'true_label' in uncertainty_df.columns:
        # 3. Enhanced Visualization - Split boxplot showing correct vs incorrect
        plt.figure(figsize=(18, 12))
        sns.boxplot(data=uncertainty_df, x='predictive_entropy', y='true_label', hue='is_correct', 
                   orient='h', order=sorted(uncertainty_df['true_label'].unique()))
        plt.title(f'Uncertainty by Class (Correct vs. Incorrect Predictions){filename_suffix}', fontsize=16)
        plt.xlabel('Predictive Entropy', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.legend(title='Is Correct')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        uncertainty_split_boxplot_filename = f"mcatcn_uncertainty_split_boxplot{filename_suffix}.png"
        plt.savefig(uncertainty_split_boxplot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Split uncertainty boxplot saved to: {uncertainty_split_boxplot_filename}")
        
        # Original uncertainty boxplot by class
        plt.figure(figsize=(18, 10))
        sorted_true_labels = sorted(uncertainty_df['true_label'].unique())
        sns.boxplot(data=uncertainty_df, x='predictive_entropy', y='true_label', orient='h', order=sorted_true_labels)
        plt.title(f'Prediction Uncertainty (Predictive Entropy) by True Class{filename_suffix}', fontsize=16)
        plt.xlabel('Predictive Entropy', fontsize=12); plt.ylabel('True Class', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
        uncertainty_boxplot_filename = f"mcatcn_uncertainty_boxplot{filename_suffix}.png" # MCA-TCN CHANGE
        plt.savefig(uncertainty_boxplot_filename, dpi=300); plt.close()
        print(f"Uncertainty boxplot saved to: {uncertainty_boxplot_filename}")

        plt.figure(figsize=(12, 7))
        sns.kdeplot(data=uncertainty_df, x='predictive_entropy', hue='is_correct', fill=True, common_norm=False, palette='crest')
        plt.title(f'Distribution of Uncertainty (Correct vs. Incorrect){filename_suffix}', fontsize=16)
        plt.xlabel('Predictive Entropy', fontsize=12); plt.ylabel('Density', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
        uncertainty_kde_filename = f"mcatcn_uncertainty_kde{filename_suffix}.png" # MCA-TCN CHANGE
        plt.savefig(uncertainty_kde_filename, dpi=300); plt.close()
        print(f"Uncertainty KDE plot saved to: {uncertainty_kde_filename}")
    else:
        print("Skipping uncertainty plot generation due to empty DataFrame or missing columns.")
    
    model.eval() # Ensure model is back in eval mode

def get_sensor_channel_mapping(config, project_root):
    """
    Extract sensor channel mapping from the feature extraction configuration.
    Returns a list of sensor names corresponding to each feature channel.
    Takes into account any excluded sensors from the configuration.
    """
    try:
        # Try to load sensor mapping from feature extraction metadata
        intermediate_dir = os.path.join(project_root, config.get('intermediate_feature_dir', 'features'))
        
        # Look for feature names file (commonly saved during feature extraction)
        feature_names_path = os.path.join(intermediate_dir, "feature_names.npy")
        if os.path.exists(feature_names_path):
            feature_names = np.load(feature_names_path, allow_pickle=True)
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()
            # Apply sensor exclusion filter
            return apply_sensor_exclusion_to_names(feature_names, config)
        
        # Alternative: look for feature metadata file
        feature_metadata_path = os.path.join(intermediate_dir, "feature_metadata.pkl")
        if os.path.exists(feature_metadata_path):
            import pickle
            with open(feature_metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            if 'feature_names' in metadata:
                # Apply sensor exclusion filter
                return apply_sensor_exclusion_to_names(metadata['feature_names'], config)
            elif 'sensor_names' in metadata:
                # Apply sensor exclusion filter
                return apply_sensor_exclusion_to_names(metadata['sensor_names'], config)
        
        # Try to infer from config
        sensor_config = config.get('sensors', {})
        if sensor_config:
            sensor_names = []
            for sensor_name, sensor_info in sensor_config.items():
                if isinstance(sensor_info, dict) and 'channels' in sensor_info:
                    channels = sensor_info['channels']
                    if isinstance(channels, list):
                        for channel in channels:
                            sensor_names.append(f"{sensor_name}_{channel}")
                    else:
                        sensor_names.append(sensor_name)
            if sensor_names:
                # Apply sensor exclusion filter
                return apply_sensor_exclusion_to_names(sensor_names, config)
        
        # Check for sensor_columns_original in config (common in OutSense projects)
        sensor_columns = config.get('sensor_columns_original', [])
        if sensor_columns and isinstance(sensor_columns, list):
            print(f"Found sensor_columns_original with {len(sensor_columns)} sensors")
            # Apply sensor exclusion filter
            return apply_sensor_exclusion_to_names(sensor_columns, config)
        
        # Fallback: check if there's a standard sensor layout
        common_sensor_patterns = [
            # Common IMU pattern (accelerometer + gyroscope)
            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
            # Extended IMU with magnetometer
            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'],
            # Multiple body locations
            ['chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
             'wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z', 'wrist_gyro_x', 'wrist_gyro_y', 'wrist_gyro_z'],
        ]
        
        print("INFO: No sensor mapping found. Using generic channel names.")
        return None
        
    except Exception as e:
        print(f"Warning: Could not load sensor mapping: {e}")
        return None

def apply_sensor_exclusion_to_names(sensor_names, config):
    """
    Apply sensor exclusion filter to a list of sensor names.
    Returns the filtered list with excluded sensors removed.
    
    Args:
        sensor_names: List of sensor names
        config: Configuration dictionary containing excluded_sensors
    
    Returns:
        Filtered list of sensor names
    """
    excluded_sensors = config.get('excluded_sensors', [])
    if not excluded_sensors:
        return sensor_names
    
    if not sensor_names:
        return sensor_names
    
    # Filter out excluded sensors
    filtered_names = [name for name in sensor_names if name not in excluded_sensors]
    
    excluded_count = len(sensor_names) - len(filtered_names)
    if excluded_count > 0:
        print(f"Applied sensor exclusion to sensor names: removed {excluded_count} sensors")
        print(f"Excluded sensors: {[name for name in sensor_names if name in excluded_sensors]}")
    
    return filtered_names

def plot_training_curves(train_losses, val_losses, test_losses, filename_suffix=""):
    """
    Plot training curves showing train, validation, and test losses.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (may contain None values)
        test_losses: List of test losses per epoch (may contain None values)
        filename_suffix: Suffix to add to saved plot filenames
    """
    # Filter out None values and create corresponding epoch indices
    epochs = range(1, len(train_losses) + 1)
    
    # Clean validation losses
    val_epochs = []
    val_clean = []
    for i, loss in enumerate(val_losses):
        if loss is not None:
            val_epochs.append(i + 1)
            val_clean.append(loss)
    
    # Clean test losses
    test_epochs = []
    test_clean = []
    for i, loss in enumerate(test_losses):
        if loss is not None:
            test_epochs.append(i + 1)
            test_clean.append(loss)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: All losses together
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    if val_clean:
        plt.plot(val_epochs, val_clean, label='Validation Loss', color='orange', linewidth=2)
    if test_clean:
        plt.plot(test_epochs, test_clean, label='Test Loss', color='green', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - All Losses{filename_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Validation vs Test comparison (if both available)
    if val_clean and test_clean:
        plt.subplot(1, 3, 2)
        plt.plot(val_epochs, val_clean, label='Validation Loss', color='orange', linewidth=2)
        plt.plot(test_epochs, test_clean, label='Test Loss', color='green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Validation vs Test Loss{filename_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Training loss focus
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Progress{filename_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    curves_filename = f"mcatcn_training_curves{filename_suffix}.png"
    plt.savefig(curves_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {curves_filename}")
    
    # Print summary statistics
    print(f"\\n--- Training Curves Summary {filename_suffix} ---")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if val_clean:
        print(f"Final validation loss: {val_clean[-1]:.4f}")
        best_val_epoch = val_epochs[np.argmin(val_clean)]
        print(f"Best validation loss: {min(val_clean):.4f} at epoch {best_val_epoch}")
    if test_clean:
        print(f"Final test loss: {test_clean[-1]:.4f}")
        if val_clean:
            # Check for overfitting indicators
            final_gap = test_clean[-1] - val_clean[-1]
            print(f"Final test-validation gap: {final_gap:.4f}")
            if final_gap > 0.1:
                print("  Warning: Large test-validation gap may indicate overfitting")
            elif final_gap < -0.1:
                print("  Warning: Test loss lower than validation loss - check data splits")

def visualize_attention_weights(model, test_loader, label_encoder, device, config=None, project_root=None, filename_suffix="", num_samples_to_visualize=50):
    """
    Visualize attention weights for interpretability of the MCA-TCN model.
    Shows which channels (sensors) the model considers most important for predictions.
    """
    print(f"\\n--- Visualizing Attention Weights for Interpretability {filename_suffix} ---")
    model.eval()
    
    # Get sensor channel mapping if available
    sensor_names = None
    if config is not None and project_root is not None:
        sensor_names = get_sensor_channel_mapping(config, project_root)
    
    all_attention_weights = []
    all_predictions = []
    all_true_labels = []
    all_correct_predictions = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            if sample_count >= num_samples_to_visualize:
                break
                
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Get predictions and attention weights
            outputs, attention_weights = model.forward_with_attention(batch_X)
            preds = torch.argmax(outputs, dim=1)
            
            # Store data for visualization
            batch_size = min(batch_X.size(0), num_samples_to_visualize - sample_count)
            all_attention_weights.append(attention_weights[:batch_size].cpu().numpy())
            all_predictions.extend(preds[:batch_size].cpu().numpy())
            all_true_labels.extend(batch_y[:batch_size].cpu().numpy())
            all_correct_predictions.extend((preds[:batch_size] == batch_y[:batch_size]).cpu().numpy())
            
            sample_count += batch_size
    
    if not all_attention_weights:
        print("No samples available for attention visualization.")
        return
    
    # Concatenate all attention weights
    all_attention_weights = np.concatenate(all_attention_weights, axis=0)
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_correct_predictions = np.array(all_correct_predictions)
    
    num_channels = all_attention_weights.shape[1]
    
    # Create channel labels (sensor names or generic)
    if sensor_names and len(sensor_names) >= num_channels:
        channel_labels = sensor_names[:num_channels]
        print(f"Using sensor names: {channel_labels[:5]}{'...' if len(channel_labels) > 5 else ''}")
    else:
        channel_labels = [f'Ch{i+1}' for i in range(num_channels)]
        if sensor_names:
            print(f"Warning: Sensor names available ({len(sensor_names)}) but don't match channel count ({num_channels})")
        print(f"Using generic channel names: {channel_labels[:5]}{'...' if len(channel_labels) > 5 else ''}")
    
    print(f"Visualizing attention weights for {len(all_attention_weights)} samples with {num_channels} channels")
    
    # 1. Average attention weights by predicted class
    class_names = label_encoder.classes_
    plt.figure(figsize=(max(15, num_channels * 0.8), max(8, len(class_names) * 0.5)))
    
    class_attention_means = []
    class_labels_for_plot = []
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = all_predictions == class_idx
        if np.sum(class_mask) > 0:
            class_attention_mean = np.mean(all_attention_weights[class_mask], axis=0)
            class_attention_means.append(class_attention_mean)
            class_labels_for_plot.append(class_name)
    
    if class_attention_means:
        class_attention_means = np.array(class_attention_means)
        
        # Create heatmap with sensor names
        ax = sns.heatmap(class_attention_means, 
                        xticklabels=channel_labels,
                        yticklabels=class_labels_for_plot,
                        annot=True, fmt='.3f', cmap='YlOrRd', cbar=True)
        
        plt.title(f'Average Attention Weights by Predicted Class{filename_suffix}', fontsize=14)
        plt.xlabel('Sensor Channel', fontsize=12)
        plt.ylabel('Predicted Class', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        attention_heatmap_filename = f"mcatcn_attention_heatmap_by_class{filename_suffix}.png"
        plt.savefig(attention_heatmap_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class-wise attention heatmap saved to: {attention_heatmap_filename}")
    
    # 2. Comparison of attention weights for correct vs incorrect predictions
    if len(np.unique(all_correct_predictions)) > 1:
        plt.figure(figsize=(max(12, num_channels * 0.6), 6))
        
        correct_attention = all_attention_weights[all_correct_predictions]
        incorrect_attention = all_attention_weights[~all_correct_predictions]
        
        channel_indices = np.arange(num_channels)
        correct_mean = np.mean(correct_attention, axis=0)
        incorrect_mean = np.mean(incorrect_attention, axis=0)
        correct_std = np.std(correct_attention, axis=0)
        incorrect_std = np.std(incorrect_attention, axis=0)
        
        plt.errorbar(channel_indices, correct_mean, yerr=correct_std, 
                    label=f'Correct Predictions (n={len(correct_attention)})', 
                    marker='o', capsize=5, capthick=2)
        plt.errorbar(channel_indices, incorrect_mean, yerr=incorrect_std, 
                    label=f'Incorrect Predictions (n={len(incorrect_attention)})', 
                    marker='s', capsize=5, capthick=2)
        
        plt.xlabel('Sensor Channel', fontsize=12)
        plt.ylabel('Average Attention Weight', fontsize=12)
        plt.title(f'Attention Weights: Correct vs Incorrect Predictions{filename_suffix}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(channel_indices, channel_labels, rotation=45, ha='right')
        plt.tight_layout()
        
        attention_correctness_filename = f"mcatcn_attention_correct_vs_incorrect{filename_suffix}.png"
        plt.savefig(attention_correctness_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correctness comparison plot saved to: {attention_correctness_filename}")
    
    # 3. Distribution of attention weights across all channels
    plt.figure(figsize=(max(12, num_channels * 0.8), 8))
    
    # Create box plot for each channel
    attention_data_for_boxplot = [all_attention_weights[:, i] for i in range(num_channels)]
    box_plot = plt.boxplot(attention_data_for_boxplot, patch_artist=True, 
                          labels=channel_labels)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, num_channels))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Sensor Channel', fontsize=12)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.title(f'Distribution of Attention Weights Across Sensor Channels{filename_suffix}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    attention_distribution_filename = f"mcatcn_attention_distribution{filename_suffix}.png"
    plt.savefig(attention_distribution_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention distribution plot saved to: {attention_distribution_filename}")
    
    # 4. Summary statistics
    print(f"\\n--- Attention Weight Statistics {filename_suffix} ---")
    print(f"Overall attention weight statistics:")
    print(f"  Mean: {np.mean(all_attention_weights):.4f}")
    print(f"  Std:  {np.std(all_attention_weights):.4f}")
    print(f"  Min:  {np.min(all_attention_weights):.4f}")
    print(f"  Max:  {np.max(all_attention_weights):.4f}")
    
    print(f"\\nPer-sensor attention weight means:")
    channel_means = np.mean(all_attention_weights, axis=0)
    for i, (mean_weight, sensor_name) in enumerate(zip(channel_means, channel_labels)):
        print(f"  {sensor_name}: {mean_weight:.4f}")
    
    # Find most and least important sensors
    most_important_idx = np.argmax(channel_means)
    least_important_idx = np.argmin(channel_means)
    print(f"\\nMost important sensor: {channel_labels[most_important_idx]} (mean weight: {channel_means[most_important_idx]:.4f})")
    print(f"Least important sensor: {channel_labels[least_important_idx]} (mean weight: {channel_means[least_important_idx]:.4f})")
    
    # Create a ranking of sensors by importance
    sensor_importance_ranking = sorted(zip(channel_labels, channel_means), key=lambda x: x[1], reverse=True)
    print(f"\\nSensor importance ranking (top 10):")
    for i, (sensor_name, importance) in enumerate(sensor_importance_ranking[:10]):
        print(f"  {i+1:2d}. {sensor_name}: {importance:.4f}")

# --- Main Execution (MODIFIED for HPO) ---
if __name__ == "__main__":
    # Determine project root and load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Basic check for project root, assuming src is a subdir of project root
    inferred_project_root = os.path.abspath(os.path.join(script_dir, '..')) if "src" in script_dir.lower() else script_dir
    
    config_path = os.path.join(inferred_project_root, CONFIG_FILE)
    if not os.path.exists(config_path):
        print(f"ERROR: Config file {config_path} not found.")
        exit()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    project_root = config.get('project_root_dir', inferred_project_root)
    print(f"Using project root: {project_root}")

    # --- ADDED: Get min_class_instances from config for undersampling ---
    min_class_instances_for_undersampling = config.get('min_class_instances', 10) # Default to 10 if not in config
    print(f"INFO: Using min_class_instances_for_undersampling = {min_class_instances_for_undersampling} for HPO folds.")
    # --- END ADDED ---

    prepared_data = prepare_data_for_tcn_hpo(config, project_root)
    if any(item is None for item in prepared_data):
        print("ERROR: Data preparation failed. Exiting.")
        exit()
    
    X_tv_all, y_tv_all, groups_tv_all, X_test_final_raw, y_test_final_raw, n_features_loaded, potential_labels = prepared_data

    if len(X_tv_all) == 0:
        print("ERROR: No data available for HPO and training. Exiting.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\\nUsing device: {device}")

    print("\\n--- Starting Hyperparameter Optimization with Optuna for MCA-TCN ---") # MCA-TCN CHANGE
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_tv_all, y_tv_all, groups_tv_all, n_features_loaded, potential_labels, device, min_class_instances_for_undersampling),
                   n_trials=NUM_OPTUNA_TRIALS, n_jobs=2) # Changed n_jobs to 1 for sequential processing

    best_params = study.best_params
    print(f"\\n--- Optuna HPO Complete ---")
    print(f"Best trial: {study.best_trial.number}, F1: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    print(f"  Learning rate: {best_params['lr']:.6f}")
    print(f"  Weight decay: {best_params['weight_decay']:.6f}")
    print(f"  Dropout rate: {best_params['dropout_rate']:.3f}")
    print(f"  TCN kernel size: {best_params['tcn_kernel_size']}")
    print(f"  Attention Reduction Ratio: {best_params['attention_reduction_ratio']}") # MCA-TCN CHANGE
    if USE_DYNAMIC_LOSS_WEIGHTING:
        print(f"  Dynamic Loss Alpha: {best_params['dynamic_loss_alpha']:.3f}")
    if USE_SELF_PACED_LEARNING:
        print(f"  SPL Decay Rate: {best_params['spl_decay_rate']:.3f}")
        print(f"  SPL Min Weight: {best_params['spl_min_weight']:.3f}")
    print(f"  Number of TCN layers: {best_params['num_tcn_layers']}")
    tcn_channels_str = [str(best_params[f'num_channels_l{i+1}']) for i in range(best_params['num_tcn_layers'])]
    print(f"  TCN channels: [{', '.join(tcn_channels_str)}]")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Classifier head type: {best_params['classifier_head_type']}")
    print(f"  Classifier dropout: {best_params['classifier_dropout']:.3f}")
    
    # Print classifier-specific parameters
    if best_params['classifier_head_type'] == 'mlp':
        print(f"  MLP hidden multiplier: {best_params['mlp_hidden_multiplier']:.3f}")
    elif best_params['classifier_head_type'] == 'deep_mlp':
        print(f"  Deep MLP layers: {best_params['deep_mlp_num_layers']}")
        print(f"  Deep MLP dimension reduction: {best_params['deep_mlp_dim_reduction']:.3f}")
    elif best_params['classifier_head_type'] == 'attention_pooling':
        print(f"  Attention hidden multiplier: {best_params['attention_hidden_multiplier']:.3f}")
    elif best_params['classifier_head_type'] == 'lstm':
        print(f"  LSTM hidden multiplier: {best_params['lstm_hidden_multiplier']:.3f}")
        print(f"  LSTM layers: {best_params['lstm_num_layers']}")
    
    print(f"  Use scheduler: {best_params['use_scheduler']}")
    if best_params['use_scheduler']:
        print(f"  Scheduler type: {best_params['scheduler_type']}")
        if best_params['scheduler_type'] == 'ReduceLROnPlateau':
            print(f"    Factor: {best_params['scheduler_factor']:.3f}")
            print(f"    Patience: {best_params['scheduler_patience']}")
            print(f"    Min LR: {best_params['scheduler_min_lr']:.2e}")
        elif best_params['scheduler_type'] == 'CosineAnnealingLR':
            print(f"    T_max: {best_params['scheduler_T_max']}")
            print(f"    Eta_min: {best_params['scheduler_eta_min']:.2e}")

    print("\\n--- Training Final MCA-TCN Model with Best Hyperparameters ---") # MCA-TCN CHANGE
    final_lr = best_params['lr']
    final_weight_decay = best_params['weight_decay']
    final_dropout = best_params['dropout_rate']
    final_tcn_kernel_size = best_params['tcn_kernel_size']
    final_batch_size = best_params['batch_size']
    final_attention_reduction = best_params['attention_reduction_ratio'] # MCA-TCN CHANGE
    final_dynamic_loss_alpha = best_params.get('dynamic_loss_alpha', DYNAMIC_LOSS_ALPHA) if USE_DYNAMIC_LOSS_WEIGHTING else DYNAMIC_LOSS_ALPHA
    final_spl_decay_rate = best_params.get('spl_decay_rate', SPL_DECAY_RATE) if USE_SELF_PACED_LEARNING else SPL_DECAY_RATE
    final_spl_min_weight = best_params.get('spl_min_weight', SPL_MIN_WEIGHT) if USE_SELF_PACED_LEARNING else SPL_MIN_WEIGHT
    
    # Extract classifier head parameters
    final_classifier_head_type = best_params['classifier_head_type']
    final_classifier_dropout = best_params['classifier_dropout']
    
    # Build classifier-specific parameters
    final_classifier_params = {}
    if final_classifier_head_type == "mlp":
        final_classifier_params["hidden_multiplier"] = best_params["mlp_hidden_multiplier"]
    elif final_classifier_head_type == "deep_mlp":
        final_classifier_params["num_layers"] = best_params["deep_mlp_num_layers"]
        final_classifier_params["dim_reduction"] = best_params["deep_mlp_dim_reduction"]
    elif final_classifier_head_type == "attention_pooling":
        final_classifier_params["attention_hidden_multiplier"] = best_params["attention_hidden_multiplier"]
    elif final_classifier_head_type == "lstm":
        final_classifier_params["lstm_hidden_multiplier"] = best_params["lstm_hidden_multiplier"]
        final_classifier_params["lstm_num_layers"] = best_params["lstm_num_layers"]
    
    # Build final TCN channels list based on tuned number of layers
    final_num_tcn_layers = best_params['num_tcn_layers']
    final_tcn_channels = []
    for layer_idx in range(final_num_tcn_layers):
        final_tcn_channels.append(best_params[f'num_channels_l{layer_idx+1}'])
    
    # Extract scheduler parameters
    final_use_scheduler = best_params['use_scheduler']
    final_scheduler_type = best_params.get('scheduler_type') if final_use_scheduler else None
    final_scheduler_params = {}
    if final_use_scheduler and final_scheduler_type:
        if final_scheduler_type == "ReduceLROnPlateau":
            final_scheduler_params = {
                "factor": best_params["scheduler_factor"],
                "patience": best_params["scheduler_patience"],
                "min_lr": best_params["scheduler_min_lr"]
            }
        elif final_scheduler_type == "CosineAnnealingLR":
            final_scheduler_params = {
                "T_max": best_params["scheduler_T_max"],
                "eta_min": best_params["scheduler_eta_min"]
            }
    
    gss_final_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    
    if len(np.unique(groups_tv_all)) < 2 and len(X_tv_all) > 0 :        
        print("Warning: Only one group for GroupShuffleSplit. Using random split for final validation.")
        from sklearn.model_selection import train_test_split
        final_train_idx, final_val_idx = train_test_split(np.arange(len(X_tv_all)), test_size=0.15, random_state=42, stratify=y_tv_all if len(np.unique(y_tv_all)) > 1 else None)
    elif len(X_tv_all) == 0:        
        print("ERROR: X_tv_all is empty before final split. Cannot proceed."); exit()
    else:
        final_train_idx, final_val_idx = next(gss_final_split.split(X_tv_all, y_tv_all, groups_tv_all))

    X_final_train_raw, X_final_val_raw = X_tv_all[final_train_idx], X_tv_all[final_val_idx]
    y_final_train_raw, y_final_val_raw = y_tv_all[final_train_idx], y_tv_all[final_val_idx]
    print(f"Final data split: Train={len(X_final_train_raw)}, Val={len(X_final_val_raw)}, Test={len(X_test_final_raw)}")

    if len(X_final_train_raw) == 0: print("ERROR: Final training set is empty."); exit()

    final_scaler = StandardScaler()
    X_final_train_scaled = final_scaler.fit_transform(X_final_train_raw.reshape(-1, n_features_loaded)).reshape(X_final_train_raw.shape)
    X_final_val_scaled = final_scaler.transform(X_final_val_raw.reshape(-1, n_features_loaded)).reshape(X_final_val_raw.shape) if len(X_final_val_raw) > 0 else np.array([])
    X_test_final_scaled = final_scaler.transform(X_test_final_raw.reshape(-1, n_features_loaded)).reshape(X_test_final_raw.shape) if len(X_test_final_raw) > 0 else np.array([])

    final_label_encoder = LabelEncoder()
    final_label_encoder.fit(potential_labels)
    y_final_train_enc = final_label_encoder.transform(y_final_train_raw)
    num_classes_final = len(final_label_encoder.classes_)
    print(f"Final model classes ({num_classes_final}): {final_label_encoder.classes_[:10]}...")
    y_final_val_enc = final_label_encoder.transform(y_final_val_raw) if len(X_final_val_raw) > 0 else np.array([])
    y_test_final_enc = final_label_encoder.transform(y_test_final_raw) if len(X_test_final_raw) > 0 else np.array([])

    final_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_final_train_scaled.transpose(0,2,1)).float(), torch.from_numpy(y_final_train_enc).long()), batch_size=final_batch_size, shuffle=True)
    final_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_final_val_scaled.transpose(0,2,1)).float(), torch.from_numpy(y_final_val_enc).long()), batch_size=final_batch_size, shuffle=False) if len(X_final_val_raw) > 0 else None
    final_test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_final_scaled.transpose(0,2,1)).float(), torch.from_numpy(y_test_final_enc).long()), batch_size=final_batch_size, shuffle=False) if len(X_test_final_raw) > 0 else None

    # MCA-TCN CHANGE: Instantiate the final modular MCATCNModel with best tuned hyperparameters
    final_tcn_model = create_mcatcn_model_with_tuned_params(
        classifier_head_type=final_classifier_head_type,
        classifier_params=final_classifier_params,
        num_inputs=n_features_loaded, 
        tcn_channels=final_tcn_channels, 
        num_classes=num_classes_final, 
        kernel_size=final_tcn_kernel_size, 
        dropout=final_dropout, 
        attention_reduction_ratio=final_attention_reduction,
        classifier_dropout=final_classifier_dropout
    )
    final_optimizer = optim.AdamW(final_tcn_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)
    # Use reduction='none' for dynamic loss weighting or SPL, otherwise use default
    final_criterion = nn.CrossEntropyLoss(reduction='none' if (USE_DYNAMIC_LOSS_WEIGHTING or USE_SELF_PACED_LEARNING) else 'mean')

    # Initialize scheduler if enabled
    final_scheduler = None
    if final_use_scheduler and final_scheduler_type:
        if final_scheduler_type == "ReduceLROnPlateau":
            final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                final_optimizer, 
                mode='min', 
                factor=final_scheduler_params["factor"],
                patience=final_scheduler_params["patience"],
                min_lr=final_scheduler_params["min_lr"],
                verbose=True
            )
        elif final_scheduler_type == "CosineAnnealingLR":
            final_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                final_optimizer,
                T_max=final_scheduler_params["T_max"],
                eta_min=final_scheduler_params["eta_min"]
            )

    model_dir_full = os.path.join(project_root, MODELS_OUTPUT_DIR)
    os.makedirs(model_dir_full, exist_ok=True)
    best_model_save_path = os.path.join(model_dir_full, MODEL_FILENAME)
    last_model_save_path = os.path.join(model_dir_full, LAST_MODEL_FILENAME)
    
    trained_best_model, saved_last_model_path, train_losses, val_losses, test_losses = train_model(
        final_tcn_model, final_train_loader, final_val_loader, final_test_loader, 
        final_optimizer, final_criterion, device, NUM_EPOCHS, 
        best_model_save_path, last_model_save_path, scheduler=final_scheduler,
        dynamic_loss_alpha=final_dynamic_loss_alpha,
        spl_decay_rate=final_spl_decay_rate,
        spl_min_weight=final_spl_min_weight
    )

    # Plot training curves with all losses
    print("\\n--- Generating Training Curves ---")
    plot_training_curves(train_losses, val_losses, test_losses, filename_suffix="_best_tuned")

    if final_test_loader and trained_best_model:
        print("\\n--- Evaluating Best Tuned MCA-TCN Model on Test Set ---") # MCA-TCN CHANGE
        evaluate_model(trained_best_model, final_test_loader, final_label_encoder, device, filename_suffix="_best_tuned")
        evaluate_with_uncertainty(trained_best_model, final_test_loader, final_label_encoder, device, filename_suffix="_best_tuned")
        # MCA-TCN CHANGE: Add attention weight visualization for interpretability with sensor mapping
        visualize_attention_weights(trained_best_model, final_test_loader, final_label_encoder, device, 
                                  config=config, project_root=project_root, filename_suffix="_best_tuned", 
                                  num_samples_to_visualize=100)
    else:
        print("Skipping final evaluation on test set (no test data or model training failed).")

    scaler_save_path = os.path.join(model_dir_full, SCALER_FILENAME)
    encoder_save_path = os.path.join(model_dir_full, ENCODER_FILENAME)
    with open(scaler_save_path, 'wb') as f: joblib.dump(final_scaler, f)
    with open(encoder_save_path, 'wb') as f: joblib.dump(final_label_encoder, f)

    print(f"\\n--- Tuned MCA-TCN Pipeline Complete ---") # MCA-TCN CHANGE
    if os.path.exists(best_model_save_path): print(f"Best tuned model: {best_model_save_path}")
    if os.path.exists(saved_last_model_path): print(f"Last tuned model state: {saved_last_model_path}")
    print(f"Tuned Scaler: {scaler_save_path}"); print(f"Tuned Label Encoder: {encoder_save_path}")

    # --- ADDED: Plot training curves for the final model ---
    plot_training_curves(train_losses, val_losses, test_losses, filename_suffix="_final_tuned")
    # --- END ADDED ---

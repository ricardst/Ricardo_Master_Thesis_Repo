# src/models.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm # For TemporalBlock

# -----------------------------
# Model Definition(s)
# -----------------------------

class Simple1DCNN(nn.Module):
    """
    A simple 1D Convolutional Neural Network for time series classification.
    Consists of multiple convolutional blocks followed by fully connected layers.
    Uses BatchNorm, Max Pooling, Dropout, ReLU activations, and Adaptive Average Pooling.
    """
    def __init__(self, input_channels, num_classes):
        """
        Initializes the layers of the CNN.

        Args:
            input_channels (int): The number of input features (channels) in the time series window.
                                  Corresponds to the number of sensors/derived features.
            num_classes (int): The number of output classes (activities).
        """
        super(Simple1DCNN, self).__init__()
        # Validate inputs
        if input_channels <= 0:
             raise ValueError(f"input_channels must be positive, got {input_channels}")
        if num_classes <= 0:
             raise ValueError(f"num_classes must be positive, got {num_classes}")

        logging.info(f"Initializing Simple1DCNN model with input_channels={input_channels}, num_classes={num_classes}")

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        # Padding='same' equivalent for kernel_size=5: padding=2
        self.bn1 = nn.BatchNorm1d(64) # Batch normalization for stabilizing training
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Max pooling to reduce sequence length
        self.dropout1 = nn.Dropout(0.3) # Dropout for regularization

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)

        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2) # Increased filter count
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.3)

        # --- Global Average Pooling ---
        # Reduces the sequence dimension to 1, making the model less sensitive to input length variations
        self.gap = nn.AdaptiveAvgPool1d(1)

        # --- Fully Connected (Dense) Layers ---
        self.fc1 = nn.Linear(256, 128) # Input size matches the out_channels of the last conv block after GAP
        self.dropout_fc1 = nn.Dropout(0.3) # Dropout before the next layer
        # Removed intermediate layers for simplicity (can be added back if needed)
        # self.fc2 = nn.Linear(128, 64)
        # self.dropout_fc2 = nn.Dropout(0.3)
        # self.fc3 = nn.Linear(64, 128)
        # self.dropout_fc3 = nn.Dropout(0.3)
        # self.fc4 = nn.Linear(128, 128)
        # self.dropout_fc4 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, num_classes) # Final output layer producing class logits

        logging.info("Simple1DCNN model layers initialized.")

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) containing raw logits.
        """
        # Input shape check (optional debug)
        # logging.debug(f"Model input shape: {x.shape}")

        # Apply Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        # logging.debug(f"After block 1 shape: {x.shape}")

        # Apply Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        # logging.debug(f"After block 2 shape: {x.shape}")

        # Apply Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        # logging.debug(f"After block 3 shape: {x.shape}")

        # Apply Global Average Pooling
        x = self.gap(x) # Shape becomes (batch_size, 256, 1)
        # logging.debug(f"After GAP shape: {x.shape}")

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Shape becomes (batch_size, 256)
        # logging.debug(f"After flatten shape: {x.shape}")

        # Apply Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x) # Apply activation
        x = self.dropout_fc1(x)
        # logging.debug(f"After FC1 shape: {x.shape}")

        # Apply final output layer (no activation here, as CrossEntropyLoss expects logits)
        x = self.fc_out(x)
        # logging.debug(f"Final output shape: {x.shape}")

        return x
    
class FEN(nn.Module):
    """Feature Extraction Network (FEN) using 1D CNNs - Processes ONE channel at a time."""
    def __init__(self, orig_in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        super(FEN, self).__init__()
        kernel_size = 3
        padding = 1
        maxpool_kernel_size = 2
        dropout_rate = 0.2

        logging.info(f"Initializing FEN (will process {orig_in_channels} channels sequentially). CNN input channel = 1.")
        # <<< MODIFIED: Conv1d in_channels is always 1 >>>
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, out_channels1, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels1, out_channels2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate)
        )
        # <<< MODIFIED: Added missing blocks based on sequential loops >>>
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(out_channels2, out_channels3, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(out_channels3, out_channels4, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x_single_channel):
        # Expects input shape: (batch, 1, seq_len)
        if x_single_channel.shape[1] != 1:
             logging.warning(f"FEN received input with {x_single_channel.shape[1]} channels, expected 1. Check model forward pass.")
             # Attempt to process first channel only as fallback? Or raise error?
             # For now, let's proceed assuming the calling code handles it.
             # raise ValueError("FEN expects input with 1 channel for sequential processing.")

        x = self.conv_block1(x_single_channel)
        x = self.conv_block2(x)
        # <<< MODIFIED: Pass through all blocks >>>
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        # <<< END MODIFIED >>>
        # Output shape: (batch, out_channels4, seq_len_out)
        # Permute is handled in the wrapper model now after concatenation
        return x

class ResBLSTM(nn.Module):
    """Residual Bidirectional LSTM Layer."""
    # ... (Keep existing ResBLSTM code - no changes needed here) ...
    def __init__(self, input_size, hidden_size, num_layers):
        super(ResBLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.transform = nn.Linear(input_size, hidden_size * 2)
        logging.debug("ResBLSTM initialized.") # Changed to debug

    def forward(self, x):
        residual = self.transform(x)
        output, (hn, cn) = self.lstm(x)
        output = self.layer_norm(output)
        return output + residual

class AttentionLayer(nn.Module):
    """Attention mechanism layer."""
    def __init__(self, input_size): # Input is the feature size from LSTM
        super(AttentionLayer, self).__init__()
        # Attention mechanism
        self.attention_weights_layer = nn.Linear(input_size, 1)
        logging.info("AttentionLayer initialized.")

    def forward(self, x):
        # x shape: (batch, seq_len, input_size which is hidden_size * 2)
        # Calculate attention scores
        attention_scores = self.attention_weights_layer(x).squeeze(-1) # -> (batch, seq_len)

        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1) # -> (batch, seq_len)

        # Calculate weighted sum of features
        # attention_weights.unsqueeze(1) -> (batch, 1, seq_len)
        # torch.bmm((batch, 1, seq_len), (batch, seq_len, input_size)) -> (batch, 1, input_size)
        weighted_feature_vector = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1) # -> (batch, input_size)

        return weighted_feature_vector, attention_weights # Return vector and weights (optional)

class FLN(nn.Module):
    """Feature Learning Network (FLN) using ResBLSTM and Attention."""
    # <<< MODIFIED: Input size depends on FEN output and number of original signals >>>
    def __init__(self, combined_fen_output_size, hidden_size, num_lstm_layers, num_classes):
        super(FLN, self).__init__()
        logging.info(f"Initializing FLN with input_size={combined_fen_output_size}, hidden={hidden_size}, classes={num_classes}")
        lstm_output_size = hidden_size * 2 # Bidirectional
        # Input size to ResBLSTM is the concatenated feature dimension
        self.res_bilstm = ResBLSTM(combined_fen_output_size, hidden_size, num_layers=num_lstm_layers)
        self.attention_layer = AttentionLayer(lstm_output_size)
        self.fc = nn.Linear(lstm_output_size, num_classes)
        logging.info("FLN initialized.")

    def forward(self, x_combined_fen):
        # x_combined_fen shape: (batch, seq_len_out, combined_fen_output_size)
        x = self.res_bilstm(x_combined_fen)    # -> (batch, seq_len_out, hidden_size * 2)
        attention_output, _ = self.attention_layer(x) # -> (batch, hidden_size * 2)
        classification_output = self.fc(attention_output) # -> (batch, num_classes)
        return classification_output

# --- Wrapper Model ---
class GeARFEN(nn.Module):
    """Combines FEN and FLN using sequential FEN processing."""

    def __init__(self, num_classes, num_original_features, # Takes num_original_features directly
                 fen_out_channels1, fen_out_channels2, fen_out_channels3, fen_out_channels4,
                 fln_hidden_size, fln_num_lstm_layers):
        super(GeARFEN, self).__init__()
        logging.info(f"Initializing GeARFEN (expects {num_original_features} features sequentially)...")
        self.num_signals = num_original_features # Store original feature count

        # FEN always takes in_channels=1 for this strategy
        self.fen = FEN(orig_in_channels=self.num_signals, # Pass original count for info log
                       out_channels1=fen_out_channels1,
                       out_channels2=fen_out_channels2,
                       out_channels3=fen_out_channels3,
                       out_channels4=fen_out_channels4)

        # FLN input size is FEN's last output channel count * number of original signals concatenated
        fln_input_size = fen_out_channels4 * self.num_signals
        self.fln = FLN(combined_fen_output_size=fln_input_size,
                       hidden_size=fln_hidden_size,
                       num_lstm_layers=fln_num_lstm_layers,
                       num_classes=num_classes)
        logging.info("GeARFEN initialized.")

    def forward(self, x):
        # x shape: (batch, num_original_features, seq_len) - IMPORTANT ASSUMPTION
        # This model's forward pass *requires* the input x to have channels == num_original_features
        batch_size, num_input_channels, seq_len = x.shape

        # Check if the input received matches the expected number of original features
        if num_input_channels != self.num_signals:
            logging.error(f"Input tensor channel dimension ({num_input_channels}) does not match the expected number of original features ({self.num_signals}) for sequential processing in GeARFEN.")
            raise ValueError(f"GeARFEN forward expects {self.num_signals} input channels, but received {num_input_channels}.")

        fen_outputs = []
        # Loop through each original signal/channel
        for i in range(self.num_signals):
            signal_input = x[:, i, :].unsqueeze(1) # (batch, 1, seq_len)
            fen_output = self.fen(signal_input) # (batch, fen_out_channels4, seq_len_out)
            fen_outputs.append(fen_output)

        # Concatenate FEN outputs along the channel dimension
        x_combined = torch.cat(fen_outputs, dim=1) # (batch, num_signals * fen_out_channels4, seq_len_out)
        # Permute for FLN/LSTM: -> (batch, seq_len_out, num_signals * fen_out_channels4)
        x_permuted = x_combined.permute(0, 2, 1)
        # Pass through FLN
        x_final = self.fln(x_permuted) # -> (batch, num_classes)
        return x_final

# --- TCN Building Blocks (adapted from tcn_activity_classification.py) ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Removes the last 'chomp_size' elements from the temporal dimension.
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Base Temporal Convolutional Network (TCN) consisting of multiple TemporalBlocks.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input x shape: (batch_size, num_features, seq_len)
        return self.network(x)

# --- New TCN-based Model Architectures ---

class MSTCN(nn.Module):
    """Multi-Scale TCN (MS-TCN)"""
    def __init__(self, num_inputs, num_classes, tcn_num_channels_list, kernel_sizes, dropout=0.2):
        super(MSTCN, self).__init__()
        logging.info(f"Initializing MS-TCN with num_inputs={num_inputs}, num_classes={num_classes}, kernel_sizes={kernel_sizes}")
        self.branches = nn.ModuleList()
        for i, kernel_size in enumerate(kernel_sizes):
            # Potentially use a different num_channels for each branch or share
            num_channels_for_branch = tcn_num_channels_list[i] if isinstance(tcn_num_channels_list[0], list) else tcn_num_channels_list
            self.branches.append(
                TemporalConvNet(num_inputs, num_channels_for_branch, kernel_size=kernel_size, dropout=dropout)
            )
        
        # Calculate the combined feature size from all branches
        # Assumes last channel count in num_channels_for_branch is the output feature count for that branch
        combined_feature_size = 0
        for i, branch in enumerate(self.branches):
            num_channels_for_branch = tcn_num_channels_list[i] if isinstance(tcn_num_channels_list[0], list) else tcn_num_channels_list
            combined_feature_size += num_channels_for_branch[-1]
            
        self.fc = nn.Linear(combined_feature_size, num_classes)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        # x shape: (batch_size, num_inputs, seq_len)
        branch_outputs = []
        for branch in self.branches:
            out = branch(x) # (batch_size, branch_out_channels, seq_len)
            branch_outputs.append(out[:, :, -1]) # Take last time step or use adaptive pooling
        
        combined = torch.cat(branch_outputs, dim=1) # (batch_size, combined_feature_size)
        return self.fc(combined)

class MBTCN(nn.Module):
    """Multi-Branch TCN (MB-TCN) - Assuming branches operate on the same full input for now"""
    def __init__(self, num_inputs, num_classes, num_branches, tcn_num_channels_per_branch, kernel_size_per_branch, dropout=0.2):
        super(MBTCN, self).__init__()
        logging.info(f"Initializing MB-TCN with num_inputs={num_inputs}, num_classes={num_classes}, num_branches={num_branches}")
        self.branches = nn.ModuleList()
        
        if not isinstance(tcn_num_channels_per_branch[0], list):
             tcn_num_channels_per_branch = [tcn_num_channels_per_branch] * num_branches
        if not isinstance(kernel_size_per_branch, list):
            kernel_size_per_branch = [kernel_size_per_branch] * num_branches

        for i in range(num_branches):
            self.branches.append(
                TemporalConvNet(num_inputs, tcn_num_channels_per_branch[i], kernel_size=kernel_size_per_branch[i], dropout=dropout)
            )
        
        combined_feature_size = sum(ch_list[-1] for ch_list in tcn_num_channels_per_branch)
        self.fc = nn.Linear(combined_feature_size, num_classes)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out[:, :, -1]) # Or adaptive pooling: F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        
        combined = torch.cat(branch_outputs, dim=1)
        return self.fc(combined)

class HybridTCNRNN(nn.Module):
    """Hybrid TCN-RNN (LSTM/GRU)"""
    def __init__(self, num_inputs, num_classes, tcn_num_channels, tcn_kernel_size, rnn_type, rnn_hidden_size, rnn_num_layers, rnn_dropout=0.2, tcn_dropout=0.2, bidirectional=True):
        super(HybridTCNRNN, self).__init__()
        logging.info(f"Initializing HybridTCNRNN with num_inputs={num_inputs}, num_classes={num_classes}, rnn_type={rnn_type}")
        self.tcn = TemporalConvNet(num_inputs, tcn_num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        tcn_output_features = tcn_num_channels[-1]

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(tcn_output_features, rnn_hidden_size, rnn_num_layers, 
                               batch_first=True, dropout=rnn_dropout, bidirectional=bidirectional)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(tcn_output_features, rnn_hidden_size, rnn_num_layers,
                              batch_first=True, dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            raise ValueError("Unsupported RNN type. Choose 'lstm' or 'gru'.")

        fc_input_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0)
        # Initialize RNN weights (optional, often default is fine)
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # x shape: (batch_size, num_inputs, seq_len)
        tcn_out = self.tcn(x) # (batch_size, tcn_output_features, seq_len)
        tcn_out_permuted = tcn_out.permute(0, 2, 1) # (batch_size, seq_len, tcn_output_features)
        
        rnn_out, _ = self.rnn(tcn_out_permuted) # (batch_size, seq_len, rnn_hidden_size * num_directions)
        
        # Use the output of the last time step
        last_time_step_out = rnn_out[:, -1, :] # (batch_size, rnn_hidden_size * num_directions)
        return self.fc(last_time_step_out)

class Attention(nn.Module):
    """Simple Scaled Dot-Product Attention layer."""
    def __init__(self, feature_dim, use_softmax=True):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.use_softmax = use_softmax
        # Learnable query, or use input itself for self-attention context
        self.query = nn.Parameter(torch.randn(feature_dim, 1)) 
        nn.init.xavier_uniform_(self.query)

    def forward(self, x_seq):
        # x_seq shape: (batch_size, seq_len, feature_dim)
        # Simplified: use a fixed query or learnable query vector
        # For self-attention, Q, K, V would be derived from x_seq
        # Here, let's do a weighted sum based on similarity to a learnable query
        
        # scores = torch.matmul(x_seq, self.query).squeeze(-1) # (batch_size, seq_len)
        # A common way for channel/feature attention on TCN output (batch, channels, seq_len)
        # If x_seq is (batch, channels, seq_len) -> permute to (batch, seq_len, channels)
        # For now, let's assume input is (batch, seq_len, features)
        
        # A simpler attention: learn weights for each feature vector in sequence
        attention_scores = torch.matmul(x_seq, self.query).squeeze(-1) # (Batch, SeqLen)
        
        if self.use_softmax:
            attention_weights = F.softmax(attention_scores, dim=1) # (Batch, SeqLen)
        else: # e.g. for some channel attention mechanisms
            attention_weights = torch.sigmoid(attention_scores)

        # Weighted sum: (Batch, SeqLen, Features) * (Batch, SeqLen, 1) -> sum over SeqLen
        weighted_sum = torch.sum(x_seq * attention_weights.unsqueeze(-1), dim=1) # (Batch, Features)
        return weighted_sum, attention_weights


class MCATCN(nn.Module):
    """TCN with Multi-Channel Attention (MCA-TCN) - Simplified: Attention over TCN output features"""
    def __init__(self, num_inputs, num_classes, tcn_num_channels, tcn_kernel_size, tcn_dropout=0.2, attention_type='channel'):
        super(MCATCN, self).__init__()
        logging.info(f"Initializing MCA-TCN with num_inputs={num_inputs}, num_classes={num_classes}, attention_type={attention_type}")
        self.tcn = TemporalConvNet(num_inputs, tcn_num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        tcn_output_features = tcn_num_channels[-1]
        self.attention_type = attention_type

        if self.attention_type == 'temporal':
            # Attention over the temporal dimension of TCN output
            self.attention = Attention(tcn_output_features) # Expects (batch, seq_len, features)
        elif self.attention_type == 'channel':
            # Attention over the channel dimension of TCN output
            # This is a common interpretation: Squeeze-and-Excitation like
            self.channel_attention_fc1 = nn.Linear(tcn_output_features, tcn_output_features // 4)
            self.channel_attention_fc2 = nn.Linear(tcn_output_features // 4, tcn_output_features)
        else:
            raise ValueError("attention_type must be 'temporal' or 'channel'")
            
        self.fc = nn.Linear(tcn_output_features, num_classes)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0)
        if self.attention_type == 'channel':
            nn.init.xavier_uniform_(self.channel_attention_fc1.weight)
            nn.init.xavier_uniform_(self.channel_attention_fc2.weight)


    def forward(self, x):
        # x shape: (batch_size, num_inputs, seq_len)
        tcn_out = self.tcn(x) # (batch_size, tcn_output_features, seq_len)

        if self.attention_type == 'temporal':
            tcn_out_permuted = tcn_out.permute(0, 2, 1) # (batch_size, seq_len, tcn_output_features)
            attended_features, _ = self.attention(tcn_out_permuted) # (batch_size, tcn_output_features)
        elif self.attention_type == 'channel':
            # Squeeze: Global Average Pooling over time
            squeeze = F.adaptive_avg_pool1d(tcn_out, 1).squeeze(-1) # (batch_size, tcn_output_features)
            # Excitation
            excitation = F.relu(self.channel_attention_fc1(squeeze))
            excitation = torch.sigmoid(self.channel_attention_fc2(excitation)) # (batch_size, tcn_output_features)
            # Apply attention: (batch, channels, seq_len) * (batch, channels, 1)
            attended_tcn_out = tcn_out * excitation.unsqueeze(-1)
            # Pool over time for classification
            attended_features = F.adaptive_avg_pool1d(attended_tcn_out, 1).squeeze(-1) # (batch_size, tcn_output_features)
        
        return self.fc(attended_features)


class SAMTCN(nn.Module):
    """TCN with Self-Attention Mechanism (SAM-TCN)"""
    def __init__(self, num_inputs, num_classes, tcn_num_channels, tcn_kernel_size, 
                 sa_num_heads, sa_hidden_dim_factor=4, sa_dropout=0.1, tcn_dropout=0.2):
        super(SAMTCN, self).__init__()
        logging.info(f"Initializing SAM-TCN with num_inputs={num_inputs}, num_classes={num_classes}")
        self.tcn = TemporalConvNet(num_inputs, tcn_num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        tcn_output_features = tcn_num_channels[-1]
        
        # Transformer Encoder Layer for self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tcn_output_features, 
            nhead=sa_num_heads,
            dim_feedforward=tcn_output_features * sa_hidden_dim_factor,
            dropout=sa_dropout,
            batch_first=True # Expects (batch, seq_len, features)
        )
        self.self_attention_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1) # Single layer for simplicity
        
        self.fc = nn.Linear(tcn_output_features, num_classes)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0)
        # TransformerEncoderLayer has its own init, often good defaults

    def forward(self, x):
        # x shape: (batch_size, num_inputs, seq_len)
        tcn_out = self.tcn(x) # (batch_size, tcn_output_features, seq_len)
        
        # Permute for TransformerEncoderLayer: (batch_size, seq_len, tcn_output_features)
        tcn_out_permuted = tcn_out.permute(0, 2, 1)
        
        sa_out = self.self_attention_encoder(tcn_out_permuted) # (batch_size, seq_len, tcn_output_features)
        
        # Use the output of the first token (like [CLS] token) or average pool
        # Here, let's average pool over the sequence length
        sa_out_pooled = sa_out.mean(dim=1) # (batch_size, tcn_output_features)
        
        return self.fc(sa_out_pooled)

# NACTCN is more complex and specific; a placeholder or simplified version:
class NACTCN(nn.Module):
    """Neighborhood Attention TCN (NAC-TCN) - Placeholder/Simplified"""
    def __init__(self, num_inputs, num_classes, tcn_num_channels, tcn_kernel_size, tcn_dropout=0.2, attention_neighborhood_size=5):
        super(NACTCN, self).__init__()
        logging.warning("NACTCN is a simplified placeholder. True NAC-TCN is more complex.")
        logging.info(f"Initializing NACTCN (Simplified) with num_inputs={num_inputs}, num_classes={num_classes}")
        self.tcn = TemporalConvNet(num_inputs, tcn_num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        tcn_output_features = tcn_num_channels[-1]
        
        # Simplified local attention: a 1D conv acting as weighted sum over a neighborhood
        self.local_attention_conv = nn.Conv1d(tcn_output_features, tcn_output_features, 
                                              kernel_size=attention_neighborhood_size, 
                                              padding=attention_neighborhood_size // 2, groups=tcn_output_features) # Depthwise-like
        self.attention_activation = nn.Softmax(dim=-1) # Apply softmax over the neighborhood scores

        self.fc = nn.Linear(tcn_output_features, num_classes)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0)
        self.local_attention_conv.weight.data.normal_(0, 0.01)


    def forward(self, x):
        # x shape: (batch_size, num_inputs, seq_len)
        tcn_out = self.tcn(x) # (batch_size, tcn_output_features, seq_len)
        
        # Simplified local attention
        # Get attention scores (weights) for each position based on its neighborhood
        attention_scores = self.local_attention_conv(tcn_out) # (batch, features, seq_len)
        attention_weights = self.attention_activation(attention_scores) # (batch, features, seq_len)
        
        # Apply attention
        attended_out = tcn_out * attention_weights # Element-wise multiplication
        
        # Pool over time for classification
        final_features = F.adaptive_avg_pool1d(attended_out, 1).squeeze(-1) # (batch_size, tcn_output_features)
        
        return self.fc(final_features)

# --- Standard TCN Model (from tcn_activity_classification.py, for completeness if used by main pipeline) ---
class BaseTCNModel(nn.Module):
    """This is the TCNModel from tcn_activity_classification.py, renamed for clarity."""
    def __init__(self, num_inputs, num_classes, num_channels_list, kernel_size=2, dropout=0.2):
        super(BaseTCNModel, self).__init__()
        logging.info(f"Initializing BaseTCNModel with num_inputs={num_inputs}, num_classes={num_classes}")
        self.tcn_core = TemporalConvNet(num_inputs, num_channels_list, kernel_size=kernel_size, dropout=dropout)
        # The TCN's output will have num_channels_list[-1] features.
        # The output sequence length depends on input sequence length and TCN structure.
        # For classification, usually a linear layer is added after pooling or taking the last output.
        self.fc = nn.Linear(num_channels_list[-1], num_classes)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        # x shape: (batch_size, num_features, seq_len)
        y = self.tcn_core(x) # Output: (batch_size, num_channels_list[-1], seq_len_out)
        # Take the output from the last time step for classification, or use adaptive pooling
        # y_pooled = F.adaptive_avg_pool1d(y, 1).squeeze(-1) # (batch_size, num_channels_list[-1])
        y_last_step = y[:, :, -1] # (batch_size, num_channels_list[-1])
        return self.fc(y_last_step)

# --- Example Usage ---
if __name__ == '__main__':
    # ... (keep existing Simple1DCNN example usage) ...

    print("\n" + "="*50)
    print("--- CNNBiLSTMAttnModel Example ---")
    logging.basicConfig(level=logging.INFO)

    # Example instantiation using parameters (mimicking config)
    input_channels_ex = 50 # Example: number of features from data_prep summary
    num_classes_ex = 12   # Example: number of activities from data_prep summary
    window_size_ex = 100  # Example: sequence length

    model_params_ex = {
        'fen_out_channels1': 64, # Example values, adjust based on config/needs
        'fen_out_channels2': 128,
        'fen_out_channels3': 256,
        'fen_out_channels4': 128,
        'fln_hidden_size': 128, # Example
        'fln_num_lstm_layers': 2
    }

    new_model_instance = GeARFEN(
        input_channels=input_channels_ex,
        num_classes=num_classes_ex,
        **model_params_ex # Unpack parameters from dict
    )
    print("\n--- New Model Architecture ---")
    print(new_model_instance)

    # Optional: Use torchinfo for a detailed summary
    try:
        from torchinfo import summary
        example_input_shape = (32, input_channels_ex, window_size_ex) # Batch=32
        print("\n--- New Model Summary (torchinfo) ---")
        summary(new_model_instance, input_size=example_input_shape[1:]) # Pass (C, L)
    except ImportError:
        print("\nInstall 'torchinfo' for a detailed model summary.")
    except Exception as e:
         print(f"\nCould not generate torchinfo summary: {e}")

    # Example forward pass
    print("\n--- New Model Example Forward Pass ---")
    dummy_input = torch.randn(32, input_channels_ex, window_size_ex)
    print(f"Dummy input shape: {dummy_input.shape}")
    try:
        new_model_instance.eval()
        with torch.no_grad():
            output = new_model_instance(dummy_input)
        print(f"Output shape: {output.shape}")
        # Output shape should be (batch_size, num_classes), e.g., (32, 12)
        assert output.shape == (32, num_classes_ex)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Error during example forward pass: {e}")
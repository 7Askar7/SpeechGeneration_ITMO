import os
import time
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import matplotlib.pyplot as plt
import numpy as np
from melbanks import LogMelFilterBanks
from thop import profile

# Dataset definition
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str, n_mels: int = 40):
        super().__init__("./", download=True)
        
        self.n_mels = n_mels
        self._walker = self._walker if subset == "training" else self._walker
        
        # Filter only "yes" and "no"
        def load_list(filename: str) -> List[str]:
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]
                
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
            
        # Filter only "yes" and "no"
        self._walker = [w for w in self._walker if os.path.basename(os.path.dirname(w)) in ['yes', 'no']]
        
        # Create mel-filters with correct parameters
        self.mel_transform = LogMelFilterBanks(
            n_mels=n_mels,
            return_complex=False,
            norm_mel='slaney'
        )
        
        # Fixed length of spectrogram (number of frames)
        self.fixed_length = 100
        
    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int]:
        fileid = self._walker[n]
        # Fixed for Windows compatibility
        label = os.path.basename(os.path.dirname(fileid))
        
        # Using soundfile as backend for torchaudio
        waveform, sample_rate = torchaudio.load(fileid, backend="soundfile")
        
        # Convert to mel-spectrogram
        mel_spectrogram = self.mel_transform(waveform[0])
        
        # Adjust mel-spectrogram to fixed length
        current_length = mel_spectrogram.shape[1]
        
        if current_length > self.fixed_length:
            # If spectrogram is longer - truncate
            mel_spectrogram = mel_spectrogram[:, :self.fixed_length]
        elif current_length < self.fixed_length:
            # If shorter - pad with zeros
            padding = torch.zeros(self.n_mels, self.fixed_length - current_length)
            mel_spectrogram = torch.cat([mel_spectrogram, padding], dim=1)
        
        # Convert labels to numerical values
        label_idx = 1 if label == 'yes' else 0
        
        return mel_spectrogram, label_idx

# CNN model with configurable groups parameter
class SpeechCommandsModel(nn.Module):
    def __init__(self, n_mels: int = 40, groups: int = 1):
        super(SpeechCommandsModel, self).__init__()
        
        self.conv1 = nn.Conv1d(n_mels, 32, kernel_size=3, stride=1, padding=1, groups=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Adaptive pooling to handle different input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(128, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the model
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def calculate_flops(self, input_size: Optional[Tuple[int, int, int]] = None) -> int:
        """
        Calculate FLOPs for the model with given input size
        
        Args:
            input_size: Tuple of (batch_size, channels, sequence_length)
            
        Returns:
            int: Number of floating point operations
        """
        if input_size is None:
            # Use dimensions corresponding to model parameters
            input_size = (1, self.conv1.in_channels, 100)
        
        # Create tensor on CPU
        input_tensor = torch.randn(*input_size)
        
        # Move tensor to the same device as the model
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)
        
        flops, _ = profile(self, inputs=(input_tensor,))
        return flops

# Function to train the model
def train_model(n_mels: int = 40, groups: int = 1, batch_size: int = 64, epochs: int = 10) -> Dict[str, Any]:
    """
    Train and evaluate the speech commands model
    
    Args:
        n_mels: Number of mel filterbanks
        groups: Number of groups for grouped convolutions
        batch_size: Batch size for training
        epochs: Number of training epochs
        
    Returns:
        Dict containing training metrics:
            - train_losses: List of training losses per epoch
            - val_accuracies: List of validation accuracies per epoch
            - epoch_times: List of training times per epoch
            - test_accuracy: Final test accuracy
            - num_params: Number of model parameters
            - flops: Floating point operations
    """
    # Load data
    train_dataset = SubsetSC("training", n_mels=n_mels)
    val_dataset = SubsetSC("validation", n_mels=n_mels)
    test_dataset = SubsetSC("testing", n_mels=n_mels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = SpeechCommandsModel(n_mels=n_mels, groups=groups)
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for training")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU for training")
    model.to(device)
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics to track
    train_losses = []
    val_accuracies = []
    epoch_times = []
    
    # Train the model
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for mel_specs, labels in train_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for mel_specs, labels in val_loader:
                mel_specs, labels = mel_specs.to(device), labels.to(device)
                outputs = model(mel_specs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s')
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mel_specs, labels in test_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            outputs = model(mel_specs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Count parameters and FLOPs
    num_params = model.count_parameters()
    flops = model.calculate_flops()
    
    print(f'Number of parameters: {num_params}')
    print(f'FLOPs: {flops}')
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'epoch_times': epoch_times,
        'test_accuracy': test_accuracy,
        'num_params': num_params,
        'flops': flops
    }

# Experiment with different n_mels values
def experiment_n_mels() -> Dict[int, Dict[str, Any]]:
    """
    Run experiments with different numbers of mel filterbanks
    
    Returns:
        Dict mapping n_mels values to their respective training results
    """
    n_mels_values = [20, 40, 80]
    results = {}
    
    for n_mels in n_mels_values:
        print(f"\nTraining model with n_mels={n_mels}")
        results[n_mels] = train_model(n_mels=n_mels, groups=1)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Training loss plot
    plt.subplot(1, 3, 1)
    for n_mels, result in results.items():
        plt.plot(result['train_losses'], label=f'n_mels={n_mels}')
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation accuracy plot
    plt.subplot(1, 3, 2)
    for n_mels, result in results.items():
        plt.plot(result['val_accuracies'], label=f'n_mels={n_mels}')
    plt.title('Validation Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # n_mels vs test accuracy plot
    plt.subplot(1, 3, 3)
    plt.plot(n_mels_values, [results[n]['test_accuracy'] for n in n_mels_values], 'o-')
    plt.title('Test Accuracy vs n_mels')
    plt.xlabel('Number of Mel Filterbanks')
    plt.ylabel('Test Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('n_mels_experiment.png')
    plt.close()
    
    return results

# Experiment with different groups values
def experiment_groups(n_mels: int = 40) -> Dict[int, Dict[str, Any]]:
    """
    Run experiments with different numbers of convolution groups
    
    Args:
        n_mels: Number of mel filterbanks to use
        
    Returns:
        Dict mapping groups values to their respective training results
    """
    groups_values = [1, 2, 4, 8, 16]
    results = {}
    
    for groups in groups_values:
        print(f"\nTraining model with groups={groups}")
        results[groups] = train_model(n_mels=n_mels, groups=groups)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Epoch training time vs groups plot
    plt.subplot(2, 2, 1)
    avg_epoch_times = [np.mean(results[g]['epoch_times']) for g in groups_values]
    plt.plot(groups_values, avg_epoch_times, 'o-')
    plt.title('Average Epoch Training Time vs Groups')
    plt.xlabel('Groups')
    plt.ylabel('Time (s)')
    
    # Number of parameters vs groups plot
    plt.subplot(2, 2, 2)
    num_params = [results[g]['num_params'] for g in groups_values]
    plt.plot(groups_values, num_params, 'o-')
    plt.title('Number of Parameters vs Groups')
    plt.xlabel('Groups')
    plt.ylabel('Number of Parameters')
    
    # FLOPs vs groups plot
    plt.subplot(2, 2, 3)
    flops = [results[g]['flops'] for g in groups_values]
    plt.plot(groups_values, flops, 'o-')
    plt.title('FLOPs vs Groups')
    plt.xlabel('Groups')
    plt.ylabel('FLOPs')
    
    # Test accuracy vs groups plot
    plt.subplot(2, 2, 4)
    test_accuracies = [results[g]['test_accuracy'] for g in groups_values]
    plt.plot(groups_values, test_accuracies, 'o-')
    plt.title('Test Accuracy vs Groups')
    plt.xlabel('Groups')
    plt.ylabel('Test Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('groups_experiment.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Check compatibility of LogMelFilterBanks with torchaudio.transforms.MelSpectrogram
    print("Checking compatibility of LogMelFilterBanks with torchaudio.transforms.MelSpectrogram...")
    signal = torch.randn(1, 16000)
    
    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=160,
        n_mels=80
    )(signal)
    
    logmelbanks = LogMelFilterBanks(
        n_mels=80,
        hop_length=160
    )(signal)
    
    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape, \
        f"Shapes don't match: {torch.log(melspec + 1e-6).shape} vs {logmelbanks.shape}"
    
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks, rtol=1e-5, atol=1e-5), \
        "Spectrogram values don't match"
    
    print("Check passed successfully!")
    
    # First experiment with different n_mels values
    n_mels_results = experiment_n_mels()
    
    # Choose optimal n_mels value for groups experiment
    best_n_mels = max(n_mels_results.keys(), 
                      key=lambda k: n_mels_results[k]['test_accuracy'])
    print(f"\nBest n_mels value: {best_n_mels}")
    
    # Experiment with different groups values
    groups_results = experiment_groups(n_mels=best_n_mels)

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from sklearn.metrics import accuracy_score, f1_score, classification_report


def extract_features(temperature, melting_curve):
    """
    Extract statistical features from melting curve data.
    (Same as generate_prompts.py)

    Args:
        temperature: array of temperature values
        melting_curve: array of melting curve intensity values

    Returns:
        dict: dictionary of extracted features
    """
    # Convert to numpy arrays
    temp = np.array(temperature)
    curve = np.array(melting_curve)

    # Basic statistics
    mean_intensity = np.mean(curve)
    std_dev = np.std(curve)
    variance = np.var(curve)
    skewness = stats.skew(curve)
    kurtosis_val = stats.kurtosis(curve)

    # Peak detection
    peaks, properties = find_peaks(curve, prominence=0.001, distance=10)
    num_peaks = len(peaks)
    peak_temps = temp[peaks] if num_peaks > 0 else []
    peak_heights = curve[peaks] if num_peaks > 0 else []

    # Peak distance ratios (distance between consecutive peaks normalized)
    peak_distances = []
    if num_peaks > 1:
        for i in range(len(peak_temps) - 1):
            peak_distances.append(peak_temps[i+1] - peak_temps[i])
        # Normalize by total range
        total_range = peak_temps[-1] - peak_temps[0]
        peak_distance_ratios = [d / total_range for d in peak_distances] if total_range > 0 else []
    else:
        peak_distance_ratios = []

    # Peak height ratios (relative to highest peak, all peaks included)
    if num_peaks > 0:
        max_height = np.max(peak_heights)
        peak_height_ratios = [h / max_height for h in peak_heights] if max_height > 0 else []
    else:
        peak_height_ratios = []

    # Frequency analysis using FFT
    n = len(curve)
    yf = fft(curve)
    xf = fftfreq(n, temp[1] - temp[0])  # Sampling interval

    # Get positive frequencies only
    positive_freq_mask = xf > 0
    positive_freqs = xf[positive_freq_mask]
    positive_amplitudes = np.abs(yf[positive_freq_mask])

    if len(positive_amplitudes) > 0:
        dominant_freq_idx = np.argmax(positive_amplitudes)
        dominant_frequency = positive_freqs[dominant_freq_idx]
        frequency_amplitude = positive_amplitudes[dominant_freq_idx] / n
    else:
        dominant_frequency = 0.0
        frequency_amplitude = 0.0

    return {
        'mean_intensity': mean_intensity,
        'std_dev': std_dev,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis_val,
        'num_peaks': num_peaks,
        'peak_temps': peak_temps,
        'peak_distance_ratios': peak_distance_ratios,
        'peak_height_ratios': peak_height_ratios,
        'dominant_frequency': dominant_frequency,
        'frequency_amplitude': frequency_amplitude
    }


def features_dict_to_vector(features_dict, max_peaks=10):
    """
    Convert feature dictionary to fixed-size numpy vector.

    Args:
        features_dict: dictionary from extract_features()
        max_peaks: maximum number of peaks to consider

    Returns:
        numpy array: flattened feature vector
    """
    # Base features: 7 (mean, std, var, skew, kurt, dom_freq, freq_amp)
    base_features = [
        features_dict['mean_intensity'],
        features_dict['std_dev'],
        features_dict['variance'],
        features_dict['skewness'],
        features_dict['kurtosis'],
        features_dict['dominant_frequency'],
        features_dict['frequency_amplitude']
    ]

    # Peak features: pad to max_peaks
    num_peaks = features_dict['num_peaks']
    peak_temps = list(features_dict['peak_temps'])
    peak_distance_ratios = list(features_dict['peak_distance_ratios'])
    peak_height_ratios = list(features_dict['peak_height_ratios'])

    # Pad peak temperatures
    peak_temp_features = peak_temps[:max_peaks] + [0.0] * max(0, max_peaks - len(peak_temps))

    # Pad peak distance ratios (max_peaks - 1 distances between max_peaks peaks)
    peak_distance_features = peak_distance_ratios[:max_peaks-1] + [0.0] * max(0, max_peaks - 1 - len(peak_distance_ratios))

    # Pad peak height ratios
    peak_height_features = peak_height_ratios[:max_peaks] + [0.0] * max(0, max_peaks - len(peak_height_ratios))

    # Combine all features
    feature_vector = base_features + [num_peaks] + peak_temp_features + peak_distance_features + peak_height_features

    return np.array(feature_vector, dtype=np.float64)


class MeltingCurveDataset(Dataset):
    """Dataset for melting curve classification"""

    def __init__(self, csv_path, label_encoder, max_peaks=10):
        """
        Args:
            csv_path: path to CSV file (train/val/test)
            label_encoder: dict mapping species name to label index
            max_peaks: maximum number of peaks for feature padding
        """
        self.df = pd.read_csv(csv_path)
        self.label_encoder = label_encoder
        self.max_peaks = max_peaks

        # Convert string representations back to lists
        self.df["Temperature (°C)"] = self.df["Temperature (°C)"].apply(eval)
        self.df["Melting Curve Data"] = self.df["Melting Curve Data"].apply(eval)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract features (returns dict)
        features_dict = extract_features(row["Temperature (°C)"], row["Melting Curve Data"])

        # Convert to fixed-size vector
        features = features_dict_to_vector(features_dict, self.max_peaks)

        # Get label
        species = row["Species"]
        label = self.label_encoder[species]

        return torch.tensor(features, dtype=torch.float64), torch.tensor(label, dtype=torch.long)


class MLP(nn.Module):
    """Multi-layer perceptron for species classification"""

    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        """
        Args:
            input_dim: input feature dimension
            hidden_dims: list of hidden layer dimensions
            num_classes: number of output classes (65)
            dropout: dropout probability
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Get predictions
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        # Get predictions
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1, all_preds, all_labels


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    data_dir = Path("prompt3_splits_prompts_seed3")
    label_encoder_path = Path("label_encoder.json")

    # Hyperparameters
    batch_size = 32
    num_epochs = 1000
    learning_rate = 1e-7
    weight_decay = 1e-8
    max_peaks = 10  # Maximum number of peaks for feature padding

    # Model architecture
    hidden_dims = [256, 128, 64]  # 3 hidden layers
    dropout = 0.1

    # Early stopping
    patience = 100
    min_delta = 0.001

    print("=" * 70)
    print("MLP Species Classification")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Hidden layers: {hidden_dims}")
    print(f"Dropout: {dropout}")
    print(f"Max peaks: {max_peaks}")
    print("=" * 70)

    # Load label encoder
    with open(label_encoder_path, 'r') as f:
        label_encoder = json.load(f)

    num_classes = len(label_encoder)
    print(f"\nNumber of classes: {num_classes}")

    # Create datasets
    train_dataset = MeltingCurveDataset(data_dir / "train.csv", label_encoder, max_peaks)
    val_dataset = MeltingCurveDataset(data_dir / "val.csv", label_encoder, max_peaks)
    test_dataset = MeltingCurveDataset(data_dir / "test.csv", label_encoder, max_peaks)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get input dimension from first sample
    sample_features, _ = train_dataset[0]
    input_dim = len(sample_features)
    print(f"Input feature dimension: {input_dim}\n")

    # Create model
    model = MLP(input_dim, hidden_dims, num_classes, dropout).to(device)
    # Convert model to float64 to match input precision
    model = model.double()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best val loss: {best_val_loss:.4f}")
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, 'best_model.pt')
        else:
            patience_counter += 1
            print(f"  ✗ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n{'=' * 70}")
            print(f"Early stopping triggered! Best val loss: {best_val_loss:.4f}")
            print(f"{'=' * 70}")
            break

    # Load best model and evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating best model on test set...")
    print("=" * 70)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Weighted F1: {test_f1:.4f}")

    # Detailed classification report
    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70)

    # Create reverse label encoder for species names
    reverse_label_encoder = {v: k for k, v in label_encoder.items()}
    target_names = [reverse_label_encoder[i] for i in range(num_classes)]

    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

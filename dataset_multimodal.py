import os
import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms


class TimeSeriesImageDataset(Dataset):
    """Dataset that loads both time series data and corresponding images"""

    def __init__(self, flag='train', image_dir=None, image_size=224):
        """
        Args:
            flag: 'train', 'val', 'test', or 'all'
            image_dir: Base directory (deprecated, paths are read from CSV)
            image_size: Size to resize images to (default 224 for ResNet)
        """
        self.flag = flag
        self.image_dir = None  # Not used, paths come from CSV
        self.image_size = image_size

        # Image transforms for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.__read_data__()

    def __read_data__(self):
        # Load label encoder
        with open('label_encoder.json', 'r') as f:
            species_to_idx = json.load(f)
        idx_to_species = {v: k for k, v in species_to_idx.items()}
        self.class_names = list(species_to_idx.keys())

        # Load CSV based on flag
        if self.flag == 'train':
            df = pd.read_csv('splits_prompts_seed0/train.csv')
        elif self.flag == 'test':
            df = pd.read_csv('splits_prompts_seed0/test.csv')
        elif self.flag == 'val':
            df = pd.read_csv('splits_prompts_seed0/val.csv')
        elif self.flag == 'all':
            train_df = pd.read_csv('splits_prompts_seed0/train.csv')
            val_df = pd.read_csv('splits_prompts_seed0/val.csv')
            test_df = pd.read_csv('splits_prompts_seed0/test.csv')
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Encode species to labels
        species_names = df['Species'].values
        self.labels = torch.tensor([species_to_idx[name] for name in species_names])

        # Store image paths
        self.image_paths = df['Plot Path'].values

        # Parse melting curve data (time series)
        import ast
        melting_curves = []
        for curve_str in df['Melting Curve Data']:
            curve = ast.literal_eval(curve_str)
            melting_curves.append(np.array(curve, dtype=np.float32))

        # Pad sequences to max length
        max_len = max(len(curve) for curve in melting_curves)
        padded_curves = []
        for curve in melting_curves:
            if len(curve) < max_len:
                padded = np.pad(curve, (0, max_len - len(curve)), mode='constant', constant_values=0)
            else:
                padded = curve
            padded_curves.append(padded)

        # Convert to torch tensor [num_samples, seq_len, 1]
        melting_curves = np.array(padded_curves)
        self.time_series = torch.from_numpy(melting_curves).unsqueeze(-1)

        # Set sequence properties
        self.seq_len = melting_curves.shape[1]
        self.max_seq_len = self.seq_len
        self.num_feature = 1

        # All samples in this split
        self.idx = np.arange(len(df))

    def __getitem__(self, index):
        i = self.idx[index]

        # Get time series
        seq_x_time = self.time_series[i]

        # Get label
        seq_y = self.labels[i]

        # Load image - normalize path separators for cross-platform compatibility
        img_path_str = self.image_paths[i].replace('\\', os.sep).replace('/', os.sep)
        img_path = Path(img_path_str)

        # Handle case where image doesn't exist
        if not img_path.exists():
            # Create a blank image if not found
            image = Image.new('RGB', (self.image_size, self.image_size), color='white')
            print(f"Warning: Image not found: {img_path}, using blank image")
        else:
            image = Image.open(img_path).convert('RGB')

        # Apply transforms
        image = self.transform(image)

        return seq_x_time, image, seq_y

    def __len__(self):
        return len(self.idx)


class TimeSeriesOnlyDataset(Dataset):
    """Original dataset that only loads time series (for backward compatibility)"""

    def __init__(self, flag='train'):
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        with open('label_encoder.json', 'r') as f:
            species_to_idx = json.load(f)
        idx_to_species = {v: k for k, v in species_to_idx.items()}
        self.class_names = list(species_to_idx.keys())

        if self.flag == 'train':
            df = pd.read_csv('splits_prompts_seed0/train.csv')
        elif self.flag == 'test':
            df = pd.read_csv('splits_prompts_seed0/test.csv')
        elif self.flag == 'val':
            df = pd.read_csv('splits_prompts_seed0/val.csv')
        elif self.flag == 'all':
            train_df = pd.read_csv('splits_prompts_seed0/train.csv')
            val_df = pd.read_csv('splits_prompts_seed0/val.csv')
            test_df = pd.read_csv('splits_prompts_seed0/test.csv')
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Encode species to labels
        species_names = df['Species'].values
        self.labels = torch.tensor([species_to_idx[name] for name in species_names])

        # Parse melting curve data
        import ast
        melting_curves = []
        for curve_str in df['Melting Curve Data']:
            curve = ast.literal_eval(curve_str)
            melting_curves.append(np.array(curve, dtype=np.float32))

        # Pad sequences to max length
        max_len = max(len(curve) for curve in melting_curves)
        padded_curves = []
        for curve in melting_curves:
            if len(curve) < max_len:
                padded = np.pad(curve, (0, max_len - len(curve)), mode='constant', constant_values=0)
            else:
                padded = curve
            padded_curves.append(padded)

        # Convert to torch tensor [num_samples, seq_len, 1]
        melting_curves = np.array(padded_curves)
        self.time_series = torch.from_numpy(melting_curves).unsqueeze(-1)

        # Set sequence properties
        self.seq_len = melting_curves.shape[1]
        self.max_seq_len = self.seq_len
        self.num_feature = 1

        # All samples in this split
        self.idx = np.arange(len(df))

    def __getitem__(self, index):
        i = self.idx[index]
        seq_x_time = self.time_series[i]
        seq_y = self.labels[i]
        return seq_x_time, seq_y

    def __len__(self):
        return len(self.idx)


# For backward compatibility
TimeSeriesDataset = TimeSeriesOnlyDataset
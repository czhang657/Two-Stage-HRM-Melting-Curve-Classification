import os
import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import AutoTokenizer

class TimeSeriesDataset(Dataset):
    def __init__(self, flag='train', use_text=False, max_text_length=512):
        self.flag = flag
        self.use_text = use_text
        self.max_text_length = max_text_length
        if self.use_text:
            self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        self.__read_data__()

    def __read_data__(self):
        with open('label_encoder.json', 'r') as f:
            species_to_idx = json.load(f)
        idx_to_species = {v: k for k, v in species_to_idx.items()}
        self.class_names = list(species_to_idx.keys())
        if self.flag == 'train':
            df = pd.read_csv('prompt3_splits_prompts_seed8/train.csv')
        elif self.flag == 'test':
            df = pd.read_csv('prompt3_splits_prompts_seed8/test.csv')
        elif self.flag == 'val':
            df = pd.read_csv('prompt3_splits_prompts_seed8/val.csv')
        elif self.flag == 'all':
            train_df = pd.read_csv('prompt3_splits_prompts_seed8/train.csv')
            val_df = pd.read_csv('prompt3_splits_prompts_seed8/val.csv')
            test_df = pd.read_csv('prompt3_splits_prompts_seed8/test.csv')
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Encode species to labels using label_encoder
        species_names = df['Species'].values
        self.labels = torch.tensor([species_to_idx[name] for name in species_names])

        # Store text prompts if using text
        # if self.use_text:
        #     self.text_prompts = df['Text_Summary_Stage1'].values
        if self.use_text:
            self.text_prompts = df['Prompt'].values
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

        # Get time series and label
        seq_x_time = self.time_series[i]
        seq_y = self.labels[i]

        if self.use_text:
            # Tokenize text prompt
            text = self.text_prompts[i]
            encoded_text = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
                return_tensors='pt'
            )
            # Remove batch dimension added by return_tensors='pt'
            encoded_text = {k: v.squeeze(0) for k, v in encoded_text.items()}
            return seq_x_time, encoded_text, seq_y
        else:
            return seq_x_time, seq_y

    def __len__(self):
        return len(self.idx)
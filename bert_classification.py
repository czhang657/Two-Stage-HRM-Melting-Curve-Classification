import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from utils import set_seed


class GatedMultiHeadAttention(nn.Module):
    """
    Qwen-style Gated Multi-Head Attention with head-specific gates.
    Gate is dynamically computed from input: gate = sigmoid(W_g @ x)
    Each head has its own gate projection.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Head-specific gate projections: each head computes its own gate from input
        # Projects from hidden_size to num_heads (one scalar gate per head)
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Compute dynamic gates from input: (batch, seq_len, num_heads)
        gates = torch.sigmoid(self.gate_proj(hidden_states))
        # Reshape to (batch, num_heads, seq_len, 1) for broadcasting
        gates = gates.transpose(1, 2).unsqueeze(-1)

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)

        # Apply head-specific dynamic gates
        # attn_output: (batch, num_heads, seq_len, head_dim)
        # gates: (batch, num_heads, seq_len, 1)
        attn_output = attn_output * gates

        # Reshape back to (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network (SwiGLU-style as used in Qwen).
    """
    def __init__(self, hidden_size, intermediate_size=None, dropout=0.1):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: swish(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class GatedTransformerLayer(nn.Module):
    """
    Transformer layer with Gated Multi-Head Attention and Gated FFN.
    """
    def __init__(self, hidden_size, num_heads, intermediate_size=None, dropout=0.1):
        super().__init__()
        self.attention = GatedMultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn = GatedFFN(hidden_size, intermediate_size, dropout)

        # Layer normalization (pre-norm style like Qwen)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        # Pre-norm attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + self.dropout(attn_output)

        # Pre-norm FFN with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output

        return hidden_states


class GatedTransformerEncoder(nn.Module):
    """
    Stack of Gated Transformer Layers that replaces BERT's encoder.
    """
    def __init__(self, hidden_size, num_heads, num_layers, intermediate_size=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedTransformerLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.final_ln(hidden_states)
        return hidden_states

class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, species2label, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.texts = self.df['Prompt'].tolist()
        # self.texts = self.df['Text_Summary_Stage1'].tolist()

        self.max_len = max_len
        self.labels = [species2label[label] for label in self.df['Species']]
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class GatedBERTClassifier(nn.Module):
    """
    BERT-based classifier with Qwen-style Gated Attention layers.
    Uses BERT's embeddings but replaces the transformer encoder with
    GatedTransformerEncoder that has head-specific gates.
    """
    def __init__(self, num_classes=55, hidden_size=768, num_heads=12, num_layers=12,
                 intermediate_size=3072, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device

        # Load BERT and keep only the embeddings
        bert = AutoModel.from_pretrained("google-bert/bert-base-cased")
        self.embeddings = bert.embeddings

        # Replace BERT's encoder with Gated Transformer Encoder
        self.encoder = GatedTransformerEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get embeddings from BERT's embedding layer
        hidden_states = self.embeddings(input_ids)

        # Pass through Gated Transformer Encoder
        hidden_states = self.encoder(hidden_states, attention_mask)

        # Use [CLS] token representation for classification
        cls_embs = hidden_states[:, 0, :]
        output = self.fc(self.dropout(cls_embs))
        return output


class BERTClassifier(nn.Module):
    """Original BERT classifier (kept for reference)."""
    def __init__(self, num_classes=55, device='cuda'):
        super().__init__()
        self.bert = AutoModel.from_pretrained("google-bert/bert-base-cased")
        self.dropout = nn.Dropout(0.2)
        self.device = device
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embs = output.last_hidden_state[:, 0, :]
        output = self.fc(self.dropout(cls_embs))
        return output

def evaluate(model, loader, criterion, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    return avg_loss, acc, f1

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, use_gated_attn=True, seed_idx=0):
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            print("New Best Model Found!")
            best_model_state = model.state_dict().copy()
            best_val_loss = val_loss

    model.load_state_dict(best_model_state)
    model_suffix = "_gated" if use_gated_attn else ""
    os.makedirs("bert_checkpoints", exist_ok=True)
    torch.save(best_model_state, f"bert_checkpoints/old_prompt_best_model_final_{seed_idx}{model_suffix}.pt")
    return model

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 65
    batch_size = 8
    epochs = 100
    lr = 2e-5
    max_len = 512
    use_gated_attn = False  # Set to False to use original BERT

    log_dir = 'old_bert_classification_logs'
    os.makedirs(log_dir, exist_ok=True)

    with open('label_encoder.json', 'r') as f:
        species2label = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    for seed_idx in range(10):
        print(f"\n{'='*60}")
        print(f"Running seed {seed_idx}")
        print(f"{'='*60}")

        # train_path = f'prompt3_splits_prompts_seed{seed_idx}/train.csv'
        # val_path = f'prompt3_splits_prompts_seed{seed_idx}/val.csv'
        # test_path = f'prompt3_splits_prompts_seed{seed_idx}/test.csv'
        train_path = f'species_prompts_original/splits_prompts_seed{seed_idx}/train.csv'
        val_path = f'species_prompts_original/splits_prompts_seed{seed_idx}/val.csv'
        test_path = f'species_prompts_original/splits_prompts_seed{seed_idx}/test.csv'
        train_dataset = TextDataset(train_path, tokenizer, species2label, max_len)
        val_dataset = TextDataset(val_path, tokenizer, species2label, max_len)
        test_dataset = TextDataset(test_path, tokenizer, species2label, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        print(batch_size)
        print(len(train_loader), len(val_loader), len(test_loader))

        if use_gated_attn:
            model = GatedBERTClassifier(num_classes=num_classes, device=device).to(device)
            print("Using GatedBERTClassifier (Qwen-style Gated Attention)")
        else:
            model = BERTClassifier(num_classes=num_classes, device=device).to(device)
            print("Using original BERTClassifier")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_model = train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=epochs, use_gated_attn=use_gated_attn, seed_idx=seed_idx)

        avg_loss, test_acc, test_f1 = evaluate(best_model, test_loader, criterion, device)
        print(f"Seed {seed_idx} - Test Accuracy: {test_acc:.4f}")
        print(f"Seed {seed_idx} - Test F1 (macro): {test_f1:.4f}")

        log_path = os.path.join(log_dir, f'seed{seed_idx}_log.txt')
        with open(log_path, 'w') as log_f:
            log_f.write(f"Seed: {seed_idx}\n")
            log_f.write(f"Model: {'GatedBERTClassifier' if use_gated_attn else 'BERTClassifier'}\n")
            log_f.write(f"Epochs: {epochs}\n")
            log_f.write(f"Batch Size: {batch_size}\n")
            log_f.write(f"Learning Rate: {lr}\n")
            log_f.write(f"Max Length: {max_len}\n")
            log_f.write(f"Num Classes: {num_classes}\n")
            log_f.write(f"Train Path: {train_path}\n")
            log_f.write(f"Val Path: {val_path}\n")
            log_f.write(f"Test Path: {test_path}\n")
            log_f.write(f"\nTest Loss: {avg_loss:.4f}\n")
            log_f.write(f"Test Accuracy: {test_acc:.4f}\n")
            log_f.write(f"Test F1 (macro): {test_f1:.4f}\n")
        print(f"Log saved to {log_path}")
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import set_seed

class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, species2label, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.texts = self.df['Prompt'].tolist()
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
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }

class BERTClassifier(nn.Module):
    def __init__(self, num_classes=55, device='cuda'):
        super().__init__()
        self.bert = AutoModel.from_pretrained("google-bert/bert-base-cased")
        self.dropout = nn.Dropout(0.2)
        self.device = device
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        # 获取BERT的输出，包括注意力权重
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        cls_embs = output.last_hidden_state[:, 0, :]
        classifier_output = self.fc(self.dropout(cls_embs))

        if output_attentions:
            return classifier_output, output.attentions
        return classifier_output

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

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10):
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
    os.makedirs("bert_checkpoints", exist_ok=True)
    torch.save(best_model_state, "bert_checkpoints/bert_with_attention_best_model.pt")
    return model

def analyze_text_lengths(loader, output_path='text_length_distribution.png'):
    """
    统计数据集中文本的实际长度分布
    """
    text_lengths = []

    for batch in tqdm(loader, desc="Analyzing text lengths"):
        attention_mask = batch["attention_mask"]
        # 计算每个样本的实际长度（非padding的token数量）
        lengths = attention_mask.sum(dim=1).cpu().numpy()
        text_lengths.extend(lengths)

    text_lengths = np.array(text_lengths)

    # 打印统计信息
    print("\n" + "="*50)
    print("Text Length Statistics:")
    print("="*50)
    print(f"Total samples: {len(text_lengths)}")
    print(f"Min length: {text_lengths.min()}")
    print(f"Max length: {text_lengths.max()}")
    print(f"Mean length: {text_lengths.mean():.2f}")
    print(f"Median length: {np.median(text_lengths):.2f}")
    print(f"Std length: {text_lengths.std():.2f}")

    # 统计各长度区间的样本数
    print("\nLength distribution by bins:")
    bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 512]
    hist, _ = np.histogram(text_lengths, bins=bins)
    for i in range(len(bins)-1):
        print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {hist[i]:4d} samples ({hist[i]/len(text_lengths)*100:5.2f}%)")

    # 统计具体长度的样本数（只显示前20个最常见的长度）
    unique_lengths, counts = np.unique(text_lengths, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    print("\nTop 20 most common text lengths:")
    for i in range(min(20, len(unique_lengths))):
        idx = sorted_indices[i]
        length = unique_lengths[idx]
        count = counts[idx]
        print(f"  Length {length:3d}: {count:4d} samples ({count/len(text_lengths)*100:5.2f}%)")

    # 绘制长度分布图
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 柱状图
    axes[0].hist(text_lengths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(text_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {text_lengths.mean():.1f}')
    axes[0].axvline(np.median(text_lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(text_lengths):.1f}')
    axes[0].set_xlabel('Text Length (number of tokens)', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Text Length Distribution (Histogram)', fontsize=14)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 累积分布图
    sorted_lengths = np.sort(text_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    axes[1].plot(sorted_lengths, cumulative, color='darkblue', linewidth=2)
    axes[1].set_xlabel('Text Length (number of tokens)', fontsize=12)
    axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1].set_title('Cumulative Text Length Distribution', fontsize=14)
    axes[1].grid(alpha=0.3)

    # 添加百分位数线
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(text_lengths, p)
        axes[1].axvline(val, color='red', linestyle='--', alpha=0.5)
        axes[1].text(val, 50, f'P{p}\n{val:.0f}', rotation=0, ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nText length distribution plot saved to {output_path}")

    return text_lengths

def analyze_attention_scores(model, loader, device, tokenizer, num_samples=100):
    """
    分析模型在推理过程中每个token位置在各层获得的平均注意力分数
    """
    model.eval()

    # 存储所有样本的注意力分数
    all_attentions = []  # List of (num_layers, num_heads, seq_len, seq_len)
    all_tokens = []
    all_attention_masks = []

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing attention scores"):
            if sample_count >= num_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 获取带注意力权重的输出
            _, attentions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

            # attentions是一个tuple，包含每一层的注意力权重
            # 每个元素的shape是 (batch_size, num_heads, seq_len, seq_len)
            batch_size = input_ids.size(0)

            for i in range(batch_size):
                if sample_count >= num_samples:
                    break

                # 收集该样本在所有层的注意力
                sample_attentions = []
                for layer_attention in attentions:
                    # layer_attention: (batch_size, num_heads, seq_len, seq_len)
                    sample_attentions.append(layer_attention[i].cpu().numpy())

                all_attentions.append(sample_attentions)

                # 解码tokens
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                all_tokens.append(tokens)
                all_attention_masks.append(attention_mask[i].cpu().numpy())

                sample_count += 1

    return all_attentions, all_tokens, all_attention_masks

def compute_average_attention_received(all_attentions, all_attention_masks):
    """
    计算每个位置在每一层平均接收到的注意力分数
    返回: (num_layers, max_seq_len) 的数组
    """
    num_samples = len(all_attentions)
    num_layers = len(all_attentions[0])
    num_heads = all_attentions[0][0].shape[0]
    seq_len = all_attentions[0][0].shape[1]

    # 存储每一层每个位置接收到的注意力总和
    layer_position_attention = np.zeros((num_layers, seq_len))
    layer_position_count = np.zeros((num_layers, seq_len))

    for sample_idx in range(num_samples):
        sample_attentions = all_attentions[sample_idx]
        attention_mask = all_attention_masks[sample_idx]

        for layer_idx in range(num_layers):
            # shape: (num_heads, seq_len, seq_len)
            layer_attn = sample_attentions[layer_idx]

            # 对所有头求平均: (seq_len, seq_len)
            avg_attn = layer_attn.mean(axis=0)

            # 计算每个位置接收到的注意力（即每列的和）
            # avg_attn[i, j] 表示位置i对位置j的注意力
            # 所以位置j接收到的注意力是 avg_attn[:, j]的和
            for pos in range(seq_len):
                if attention_mask[pos] > 0:  # 只统计有效token
                    received_attention = avg_attn[:, pos].sum()
                    layer_position_attention[layer_idx, pos] += received_attention
                    layer_position_count[layer_idx, pos] += 1

    # 计算平均值
    avg_attention_per_layer = np.divide(
        layer_position_attention,
        layer_position_count,
        where=layer_position_count > 0
    )

    return avg_attention_per_layer

def plot_attention_heatmap(avg_attention_per_layer, save_path='attention_heatmap.png', max_pos=100):
    """
    绘制注意力热图：每一层的每个位置接收到的平均注意力
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 只显示前max_pos个位置
    max_pos = min(max_pos, avg_attention_per_layer.shape[1])
    data = avg_attention_per_layer[:, :max_pos]

    # 左图：热图
    im = axes[0].imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    axes[0].set_xlabel('Token Position', fontsize=12)
    axes[0].set_ylabel('Layer', fontsize=12)
    axes[0].set_title('Average Attention Scores by Position and Layer', fontsize=13)
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('Avg Attention Score', fontsize=11)

    # 设置x轴刻度
    if max_pos <= 50:
        axes[0].set_xticks(range(0, max_pos, 5))
    else:
        axes[0].set_xticks(range(0, max_pos, 10))

    # 右图：每层的平均注意力分布
    layer_means = data.mean(axis=1)
    layer_stds = data.std(axis=1)
    layers = range(len(layer_means))

    axes[1].plot(layers, layer_means, marker='o', linewidth=2, markersize=6, color='darkred')
    axes[1].fill_between(layers, layer_means - layer_stds, layer_means + layer_stds, alpha=0.3, color='red')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Average Attention Score', fontsize=12)
    axes[1].set_title('Average Attention per Layer (with std)', fontsize=13)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention heatmap saved to {save_path}")

def plot_position_attention_distribution(avg_attention_per_layer, save_path='position_attention_dist.png', max_pos=100):
    """
    绘制不同位置在所有层的平均注意力分布
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 计算每个位置在所有层的平均注意力
    max_pos = min(max_pos, avg_attention_per_layer.shape[1])
    position_avg = avg_attention_per_layer[:, :max_pos].mean(axis=0)
    position_std = avg_attention_per_layer[:, :max_pos].std(axis=0)

    # 上图：柱状图显示每个位置的平均注意力
    axes[0].bar(range(max_pos), position_avg, color='steelblue', alpha=0.7, edgecolor='navy')
    axes[0].errorbar(range(max_pos), position_avg, yerr=position_std, fmt='none', ecolor='red', alpha=0.3, capsize=2)
    axes[0].set_xlabel('Token Position', fontsize=12)
    axes[0].set_ylabel('Average Attention Score', fontsize=12)
    axes[0].set_title('Average Attention Score by Position (across all layers, with std)', fontsize=13)
    axes[0].grid(axis='y', alpha=0.3)

    # 标注前5个最高注意力的位置
    top_5_positions = np.argsort(position_avg)[::-1][:5]
    for pos in top_5_positions:
        axes[0].text(pos, position_avg[pos], f'{pos}', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    # 下图：前几层、中间层、后几层的对比
    early_layers = avg_attention_per_layer[:4, :max_pos].mean(axis=0)  # Layer 0-3
    mid_layers = avg_attention_per_layer[4:8, :max_pos].mean(axis=0)   # Layer 4-7
    late_layers = avg_attention_per_layer[8:, :max_pos].mean(axis=0)   # Layer 8-11

    axes[1].plot(range(max_pos), early_layers, label='Early Layers (0-3)', linewidth=2, alpha=0.8)
    axes[1].plot(range(max_pos), mid_layers, label='Middle Layers (4-7)', linewidth=2, alpha=0.8)
    axes[1].plot(range(max_pos), late_layers, label='Late Layers (8-11)', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Token Position', fontsize=12)
    axes[1].set_ylabel('Average Attention Score', fontsize=12)
    axes[1].set_title('Attention Score Comparison Across Layer Groups', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Position attention distribution saved to {save_path}")

def plot_layer_attention_trend(avg_attention_per_layer, positions=[0, 1, 2, 10, 20, 50], save_path='layer_attention_trend.png'):
    """
    绘制特定位置在不同层的注意力变化趋势
    """
    plt.figure(figsize=(12, 6))

    num_layers = avg_attention_per_layer.shape[0]

    for pos in positions:
        if pos < avg_attention_per_layer.shape[1]:
            plt.plot(range(num_layers), avg_attention_per_layer[:, pos], marker='o', label=f'Position {pos}')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Average Attention Score', fontsize=12)
    plt.title('Attention Score Trends Across Layers for Different Positions', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Layer attention trend saved to {save_path}")

def analyze_token_level_attention(all_attentions, all_tokens, all_attention_masks):
    """
    分析每个具体token/字符获得的平均注意力
    返回: token到注意力分数的映射
    """
    token_attention_sum = {}  # token -> total attention received
    token_count = {}  # token -> count of occurrences

    for sample_idx in range(len(all_attentions)):
        sample_attentions = all_attentions[sample_idx]
        tokens = all_tokens[sample_idx]
        attention_mask = all_attention_masks[sample_idx]

        valid_len = int(attention_mask.sum())

        # 对所有层求平均
        for layer_idx in range(len(sample_attentions)):
            layer_attn = sample_attentions[layer_idx].mean(axis=0)  # 对所有头求平均

            # 计算每个token接收到的注意力
            for pos in range(valid_len):
                token = tokens[pos]
                # 跳过特殊token如[PAD]
                if token == '[PAD]':
                    continue

                # 该token接收到的注意力是该列的和
                received_attention = layer_attn[:valid_len, pos].sum()

                if token not in token_attention_sum:
                    token_attention_sum[token] = 0
                    token_count[token] = 0

                token_attention_sum[token] += received_attention
                token_count[token] += 1

    # 计算平均注意力
    token_avg_attention = {
        token: token_attention_sum[token] / token_count[token]
        for token in token_attention_sum
    }

    return token_avg_attention, token_count

def plot_top_tokens_attention(token_avg_attention, token_count, save_path='top_tokens_attention.png', top_n=50, min_count=5):
    """
    可视化获得最高注意力的tokens
    """
    # 过滤掉出现次数太少的tokens
    filtered_tokens = {
        token: attn for token, attn in token_avg_attention.items()
        if token_count[token] >= min_count
    }

    # 排序获取top N
    sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:top_n]

    tokens = [t[0] for t in top_tokens]
    attentions = [t[1] for t in top_tokens]
    counts = [token_count[t] for t in tokens]

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # 上图：Top N tokens的平均注意力
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tokens)))
    bars = axes[0].barh(range(len(tokens)), attentions, color=colors, edgecolor='black')
    axes[0].set_yticks(range(len(tokens)))
    axes[0].set_yticklabels(tokens, fontsize=10)
    axes[0].set_xlabel('Average Attention Score', fontsize=12)
    axes[0].set_title(f'Top {top_n} Tokens with Highest Average Attention (min count: {min_count})', fontsize=14)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, attn, count) in enumerate(zip(bars, attentions, counts)):
        axes[0].text(attn, i, f' {attn:.3f} (n={count})', va='center', fontsize=8)

    # 下图：最低注意力的tokens
    bottom_tokens = sorted_tokens[-top_n:]
    bottom_tokens.reverse()  # 从低到高排序

    tokens_bottom = [t[0] for t in bottom_tokens]
    attentions_bottom = [t[1] for t in bottom_tokens]
    counts_bottom = [token_count[t] for t in tokens_bottom]

    colors_bottom = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(tokens_bottom)))
    bars_bottom = axes[1].barh(range(len(tokens_bottom)), attentions_bottom, color=colors_bottom, edgecolor='black')
    axes[1].set_yticks(range(len(tokens_bottom)))
    axes[1].set_yticklabels(tokens_bottom, fontsize=10)
    axes[1].set_xlabel('Average Attention Score', fontsize=12)
    axes[1].set_title(f'Bottom {top_n} Tokens with Lowest Average Attention (min count: {min_count})', fontsize=14)
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, attn, count) in enumerate(zip(bars_bottom, attentions_bottom, counts_bottom)):
        axes[1].text(attn, i, f' {attn:.3f} (n={count})', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top tokens attention visualization saved to {save_path}")

def plot_token_type_attention(token_avg_attention, token_count, save_path='token_type_attention.png', min_count=5):
    """
    按token类型分析注意力分布
    """
    # 分类tokens
    special_tokens = []
    punctuation_tokens = []
    word_tokens = []
    subword_tokens = []

    for token, attn in token_avg_attention.items():
        if token_count[token] < min_count:
            continue

        if token in ['[CLS]', '[SEP]', '[MASK]', '[UNK]']:
            special_tokens.append((token, attn))
        elif token in ['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '"', "'", '`']:
            punctuation_tokens.append((token, attn))
        elif token.startswith('##'):
            subword_tokens.append((token, attn))
        else:
            word_tokens.append((token, attn))

    # 计算各类别的统计信息
    categories = []
    means = []
    medians = []
    stds = []

    for name, token_list in [('Special', special_tokens), ('Punctuation', punctuation_tokens),
                              ('Words', word_tokens), ('Subwords', subword_tokens)]:
        if token_list:
            attns = [t[1] for t in token_list]
            categories.append(f'{name}\n(n={len(token_list)})')
            means.append(np.mean(attns))
            medians.append(np.median(attns))
            stds.append(np.std(attns))

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：各类别的平均注意力
    x_pos = np.arange(len(categories))
    axes[0].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['red', 'orange', 'steelblue', 'purple'], edgecolor='black')
    axes[0].scatter(x_pos, medians, color='darkred', s=100, zorder=5, marker='D', label='Median')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(categories, fontsize=11)
    axes[0].set_ylabel('Average Attention Score', fontsize=12)
    axes[0].set_title('Average Attention by Token Type', fontsize=14)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, (mean, median) in enumerate(zip(means, medians)):
        axes[0].text(i, mean, f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    # 右图：箱线图显示分布
    data_for_box = []
    labels_for_box = []
    for name, token_list in [('Special', special_tokens), ('Punctuation', punctuation_tokens),
                              ('Words', word_tokens), ('Subwords', subword_tokens)]:
        if token_list:
            attns = [t[1] for t in token_list]
            data_for_box.append(attns)
            labels_for_box.append(name)

    bp = axes[1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                          notch=True, showmeans=True)

    # 设置箱线图颜色
    colors = ['red', 'orange', 'steelblue', 'purple']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_ylabel('Attention Score', fontsize=12)
    axes[1].set_title('Attention Distribution by Token Type (Boxplot)', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Token type attention analysis saved to {save_path}")

def visualize_sample_attention(all_attentions, all_tokens, all_attention_masks, sample_idx=0, layer_idx=0, save_path='sample_attention.png'):
    """
    可视化单个样本在某一层的注意力矩阵
    """
    sample_attentions = all_attentions[sample_idx]
    tokens = all_tokens[sample_idx]
    attention_mask = all_attention_masks[sample_idx]

    # 获取有效token长度
    valid_len = int(attention_mask.sum())
    valid_tokens = tokens[:valid_len]

    # 获取指定层的注意力，对所有头求平均
    layer_attn = sample_attentions[layer_idx].mean(axis=0)  # (seq_len, seq_len)
    layer_attn = layer_attn[:valid_len, :valid_len]

    plt.figure(figsize=(max(12, valid_len * 0.3), max(10, valid_len * 0.25)))
    sns.heatmap(
        layer_attn,
        xticklabels=valid_tokens,
        yticklabels=valid_tokens,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        square=True
    )
    plt.xlabel('Key Tokens', fontsize=12)
    plt.ylabel('Query Tokens', fontsize=12)
    plt.title(f'Attention Matrix - Sample {sample_idx}, Layer {layer_idx}', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample attention visualization saved to {save_path}")

def visualize_sample_with_token_attention(all_attentions, all_tokens, all_attention_masks, sample_idx=0, save_path='sample_token_attention.png'):
    """
    可视化单个样本中每个token接收到的总注意力（跨所有层）
    """
    sample_attentions = all_attentions[sample_idx]
    tokens = all_tokens[sample_idx]
    attention_mask = all_attention_masks[sample_idx]

    valid_len = int(attention_mask.sum())
    valid_tokens = tokens[:valid_len]

    # 计算每个token在所有层接收到的平均注意力
    token_attentions = np.zeros(valid_len)

    for layer_idx in range(len(sample_attentions)):
        layer_attn = sample_attentions[layer_idx].mean(axis=0)  # (seq_len, seq_len)
        # 每个token接收到的注意力是该列的和
        for pos in range(valid_len):
            token_attentions[pos] += layer_attn[:valid_len, pos].sum()

    # 对层数求平均
    token_attentions /= len(sample_attentions)

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(max(16, valid_len * 0.3), 10))

    # 上图：柱状图
    colors = plt.cm.RdYlGn_r(token_attentions / token_attentions.max())
    axes[0].bar(range(valid_len), token_attentions, color=colors, edgecolor='black')
    axes[0].set_xticks(range(valid_len))
    axes[0].set_xticklabels(valid_tokens, rotation=90, fontsize=9)
    axes[0].set_ylabel('Average Attention Received', fontsize=12)
    axes[0].set_title(f'Token-level Attention Analysis - Sample {sample_idx}', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)

    # 标注前5个最高注意力的token
    top_5_indices = np.argsort(token_attentions)[::-1][:5]
    for idx in top_5_indices:
        axes[0].text(idx, token_attentions[idx], f'{token_attentions[idx]:.2f}',
                     ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    # 下图：热图显示每个token在每一层的注意力
    layer_token_attentions = np.zeros((len(sample_attentions), valid_len))
    for layer_idx in range(len(sample_attentions)):
        layer_attn = sample_attentions[layer_idx].mean(axis=0)
        for pos in range(valid_len):
            layer_token_attentions[layer_idx, pos] = layer_attn[:valid_len, pos].sum()

    im = axes[1].imshow(layer_token_attentions, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    axes[1].set_yticks(range(len(sample_attentions)))
    axes[1].set_ylabel('Layer', fontsize=12)
    axes[1].set_xticks(range(valid_len))
    axes[1].set_xticklabels(valid_tokens, rotation=90, fontsize=9)
    axes[1].set_xlabel('Token', fontsize=12)
    axes[1].set_title('Attention Received by Each Token Across Layers', fontsize=13)
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Attention Score', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample token-level attention saved to {save_path}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_classes = 65
    batch_size = 8
    max_len = 512
    model_path = 'bert_checkpoints/new_prompt_best_model_final_0.pt'

    train_path = 'prompt3_splits_prompts_seed0/train.csv'
    val_path = 'prompt3_splits_prompts_seed0/val.csv'
    test_path = 'prompt3_splits_prompts_seed0/test.csv'

    with open('label_encoder.json', 'r') as f:
        species2label = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    train_dataset = TextDataset(train_path, tokenizer, species2label, max_len)
    val_dataset = TextDataset(val_path, tokenizer, species2label, max_len)
    test_dataset = TextDataset(test_path, tokenizer, species2label, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # 加载预训练模型
    print("\n" + "="*50)
    print(f"Loading model from {model_path}...")
    print("="*50)
    best_model = BERTClassifier(num_classes).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    best_model.eval()
    print("Model loaded successfully.")

    # 评估模型
    print("\n" + "="*50)
    print("Evaluating on Test Set...")
    print("="*50)
    criterion = nn.CrossEntropyLoss()
    avg_loss, test_acc, test_f1 = evaluate(best_model, test_loader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")

    # 创建输出目录
    output_dir = 'attention_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # 分析文本长度分布
    print("\n" + "="*50)
    print("Analyzing Text Length Distribution...")
    print("="*50)

    print("\nTrain set:")
    train_lengths = analyze_text_lengths(train_loader, output_path=os.path.join(output_dir, 'train_text_length_distribution.png'))

    print("\nValidation set:")
    val_lengths = analyze_text_lengths(val_loader, output_path=os.path.join(output_dir, 'val_text_length_distribution.png'))

    print("\nTest set:")
    test_lengths = analyze_text_lengths(test_loader, output_path=os.path.join(output_dir, 'test_text_length_distribution.png'))

    # 分析注意力分数
    print("\n" + "="*50)
    print("Analyzing Attention Scores...")
    print("="*50)

    # 在测试集上分析注意力
    all_attentions, all_tokens, all_attention_masks = analyze_attention_scores(
        best_model, test_loader, device, tokenizer, num_samples=100
    )

    # 计算每个位置在每一层接收到的平均注意力
    avg_attention_per_layer = compute_average_attention_received(all_attentions, all_attention_masks)

    # 保存注意力数据
    np.save(os.path.join(output_dir, 'avg_attention_per_layer.npy'), avg_attention_per_layer)
    print(f"Attention data saved to {output_dir}/avg_attention_per_layer.npy")

    # 生成各种可视化
    print("\nGenerating visualizations...")

    # 根据文本长度的中位数来决定显示多少位置
    median_length = int(np.median(test_lengths))
    max_display_pos = min(median_length + 20, 150)  # 显示到中位数+20，但不超过150

    print(f"Text median length: {median_length}, displaying up to position {max_display_pos}")

    plot_attention_heatmap(
        avg_attention_per_layer,
        save_path=os.path.join(output_dir, 'attention_heatmap.png'),
        max_pos=max_display_pos
    )

    plot_position_attention_distribution(
        avg_attention_per_layer,
        save_path=os.path.join(output_dir, 'position_attention_dist.png'),
        max_pos=max_display_pos
    )

    # 选择一些关键位置来观察层间变化
    key_positions = [0, 1, 2]  # CLS和前两个token
    # 添加一些中间位置
    if median_length > 10:
        key_positions.extend([5, 10, median_length//4, median_length//2, median_length])
    plot_layer_attention_trend(
        avg_attention_per_layer,
        positions=key_positions,
        save_path=os.path.join(output_dir, 'layer_attention_trend.png')
    )

    # Token级别的注意力分析
    print("\n" + "="*50)
    print("Analyzing Token-level Attention...")
    print("="*50)

    token_avg_attention, token_count = analyze_token_level_attention(
        all_attentions, all_tokens, all_attention_masks
    )

    # 打印token级别的统计信息
    print(f"\nTotal unique tokens analyzed: {len(token_avg_attention)}")

    # 排序获取最高和最低注意力的tokens
    sorted_token_attn = sorted(token_avg_attention.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 20 tokens with HIGHEST attention:")
    for i, (token, attn) in enumerate(sorted_token_attn[:20], 1):
        print(f"  {i:2d}. '{token}': {attn:.4f} (appeared {token_count[token]} times)")

    print("\nTop 20 tokens with LOWEST attention:")
    for i, (token, attn) in enumerate(sorted_token_attn[-20:], 1):
        print(f"  {i:2d}. '{token}': {attn:.4f} (appeared {token_count[token]} times)")

    # 可视化token级别的注意力
    plot_top_tokens_attention(
        token_avg_attention, token_count,
        save_path=os.path.join(output_dir, 'top_tokens_attention.png'),
        top_n=50, min_count=5
    )

    plot_token_type_attention(
        token_avg_attention, token_count,
        save_path=os.path.join(output_dir, 'token_type_attention.png'),
        min_count=5
    )

    # 可视化几个样本的token级别注意力
    print("\nGenerating sample-specific token attention visualizations...")
    for sample_idx in range(min(5, len(all_attentions))):
        visualize_sample_with_token_attention(
            all_attentions, all_tokens, all_attention_masks,
            sample_idx=sample_idx,
            save_path=os.path.join(output_dir, f'sample_{sample_idx}_token_attention.png')
        )

    # 可视化几个样本的注意力矩阵
    print("\nGenerating sample attention matrices...")
    for sample_idx in range(min(3, len(all_attentions))):
        for layer_idx in [0, 5, 11]:  # 第一层、中间层、最后一层
            visualize_sample_attention(
                all_attentions, all_tokens, all_attention_masks,
                sample_idx=sample_idx,
                layer_idx=layer_idx,
                save_path=os.path.join(output_dir, f'sample_{sample_idx}_layer_{layer_idx}_attention.png')
            )

    print("\n" + "="*50)
    print("All visualizations completed!")
    print(f"Check the '{output_dir}' directory for all attention visualizations.")
    print("="*50)

    # 打印一些统计信息
    print("\n" + "="*50)
    print("Position-level Attention Statistics:")
    print("="*50)
    print(f"Number of layers: {avg_attention_per_layer.shape[0]}")
    print(f"Sequence length: {avg_attention_per_layer.shape[1]}")

    # 找出接收最高注意力的位置（平均所有层）
    avg_across_layers = avg_attention_per_layer.mean(axis=0)
    top_positions = np.argsort(avg_across_layers)[::-1][:10]
    print("\nTop 10 positions receiving highest attention (averaged across all layers):")
    for i, pos in enumerate(top_positions, 1):
        print(f"  {i}. Position {pos}: {avg_across_layers[pos]:.4f}")

    # 保存token级别的注意力数据
    import pickle
    with open(os.path.join(output_dir, 'token_attention_data.pkl'), 'wb') as f:
        pickle.dump({
            'token_avg_attention': token_avg_attention,
            'token_count': token_count
        }, f)
    print(f"\nToken-level attention data saved to {output_dir}/token_attention_data.pkl")

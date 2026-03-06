import os
import torch
from torch import nn
from layers.Encoder_Layer import EncoderLayer, CrossAttention
from layers.Embed import PatchEmbedding
from transformers import AutoModel


class Model(nn.Module):
    def __init__(self, args, device='cuda', patch_len=16, stride=16, use_text=False):
        super(Model, self).__init__()
        self.device = device
        self.d_model = args.d_model
        self.dropout = nn.Dropout(args.dropout)
        self.num_class = args.num_class
        self.use_text = use_text
        self.use_gate = getattr(args, 'use_gate', True)  # head-specific gate
        self.enc_in = args.enc_in  # number of variables
        padding = stride
        self.patch_embedding = PatchEmbedding(self.d_model, patch_len, stride, padding, args.dropout)

        # Load BERT model if using text
        if self.use_text:
            os.environ['TOKENIZERS_PARALLELISM'] = 'True'
            self.lm_model = AutoModel.from_pretrained("google-bert/bert-base-cased")

            # Load checkpoint
            checkpoint = torch.load("bert_checkpoints/new_prompt_best_model_final_1.pt", map_location=device)

            # Remove 'bert.' or 'module.' prefixes if present
            if any(k.startswith("bert.") for k in checkpoint.keys()):
                checkpoint = {k.replace("bert.", ""): v for k, v in checkpoint.items() if k.startswith("bert.")}

            if any(k.startswith("module.") for k in checkpoint.keys()):
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

            self.lm_model.load_state_dict(checkpoint, strict=False)

            # Project BERT embeddings (768) to d_model
            self.layer_text = nn.Linear(768, self.d_model)

        # learnable queries definition
        self.learnable_queries = nn.Parameter(torch.randn(1, 200, self.d_model))
        # self.head_nf = self.d_model * (self.learnable_queries.shape[1])
        self.head_nf = self.d_model * (218) #218, 434
        self.queries_self_attn = EncoderLayer(
            self.d_model,
            CrossAttention(d_model=self.d_model, use_gate=self.use_gate)
        )
        self.cross_attn_queries = EncoderLayer(
            self.d_model,
            CrossAttention(d_model=self.d_model, use_gate=self.use_gate)
        )
        self.cross_attn_overall = EncoderLayer(
            self.d_model,
            CrossAttention(d_model=self.d_model, use_gate=self.use_gate)
        )
        self.classification = nn.Linear(self.head_nf + 128, self.num_class)


    def forward(self, x_enc_time, x_enc_text=None):
        B = x_enc_time.size(0)
        means = x_enc_time.mean(1, keepdim=True).detach()
        x_enc_time = x_enc_time - means
        stdev = torch.sqrt(
            torch.var(x_enc_time, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_time /= stdev

        # do patching and embedding
        x_enc_time = x_enc_time.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc_time)

        # Process text if provided
        if self.use_text and x_enc_text is not None:
            # Get BERT embeddings
            outputs = self.lm_model(
                input_ids=x_enc_text['input_ids'].to(self.device),
                attention_mask=x_enc_text['attention_mask'].to(self.device),
                output_hidden_states=True
            )

            # Extract [CLS] token embedding from last hidden state
            emb_text = outputs['hidden_states'][-1][:, 0, :]  # [batch_size, 768]

            # Project to d_model
            emb_text = self.layer_text(emb_text)  # [batch_size, d_model]
            emb_text = self.dropout(emb_text)

            # Expand to match time series shape and concatenate
            # x is [batch_size * n_vars, patch_num, d_model]
            # We need to add text embedding as an additional patch
            emb_text = emb_text.unsqueeze(1).repeat(1, n_vars, 1)  # [batch_size, n_vars, d_model]
            emb_text = emb_text.view(-1, self.d_model).unsqueeze(1)  # [batch_size * n_vars, 1, d_model]

            # Concatenate text embedding with time series patches
            x = torch.cat((x, emb_text), dim=1)  # [batch_size * n_vars, patch_num + 1, d_model]

        queries, _  = self.queries_self_attn(self.learnable_queries.expand(B,-1,-1), self.learnable_queries.expand(B,-1,-1))
        # readed_curve, _ = self.cross_attn_queries(queries, x)
        readed_curve, _ = self.cross_attn_overall(x, x)
        readed_curve = readed_curve.permute(0, 2, 1)

        cls_out = readed_curve.reshape(B, -1)
        features = cls_out  # 保存特征用于contrastive loss
        output_time = self.dropout(cls_out)
        logits = self.classification(output_time)
        return logits, features



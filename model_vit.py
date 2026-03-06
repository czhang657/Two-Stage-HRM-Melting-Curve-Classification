"""
model_vit.py

Lightweight Vision Transformer (ViT) for melting curve image classification.

Architecture
------------
image (B, 3, 224, 224)
  → patch embedding (Conv2d, patch_size × patch_size)
  → (B, N, d_model)           e.g. patch_size=16 → N=196 patches
  → prepend [CLS] token
  → (B, N+1, d_model)
  → learnable positional embedding
  → n_layers MH self-attention encoder (Pre-LN, BERT-style)
  → [CLS] output  (B, d_model)
  → dropout → Linear → logits  (B, num_classes)

Small variant (1 layer, d=128, patch 16×16)
  → 196 patch tokens — mirrors the 217-token patching in model.py
  → ~260 K params (all trainable; no frozen backbone)

Large variant (3 layers, d=128, patch 16×16): ~510 K params
"""

import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    """
    Custom lightweight ViT for melting curve image classification.

    Parameters
    ----------
    num_classes : int   — output classes
    img_size    : int   — input image size (assumed square, default 224)
    patch_size  : int   — patch size in pixels (default 16)
    d_model     : int   — transformer hidden dim (default 128)
    n_heads     : int   — attention heads (default 8)
    n_layers    : int   — number of transformer encoder layers (default 1)
    d_ff        : int   — feedforward dim (default 256)
    dropout     : float — dropout rate (default 0.2)
    """

    def __init__(
        self,
        num_classes: int,
        img_size:    int   = 224,
        patch_size:  int   = 16,
        d_model:     int   = 128,
        n_heads:     int   = 8,
        n_layers:    int   = 1,
        d_ff:        int   = 256,
        dropout:     float = 0.2,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

        self.num_patches = (img_size // patch_size) ** 2   # 196 for 224/16

        # ------------------------------------------------------------------ #
        # Patch embedding: Conv2d with kernel=stride=patch_size               #
        # Equivalent to flattening each patch and projecting, but faster.     #
        # Output: (B, d_model, H/P, W/P) → flatten → (B, N, d_model)        #
        # ------------------------------------------------------------------ #
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=d_model,
            kernel_size=patch_size, stride=patch_size,
        )

        # ------------------------------------------------------------------ #
        # [CLS] token + learnable positional embedding                        #
        # ------------------------------------------------------------------ #
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed  = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model) * 0.02
        )

        # ------------------------------------------------------------------ #
        # Transformer encoder (Pre-LayerNorm = more stable for small data)   #
        # ------------------------------------------------------------------ #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------ #
        # Classification head                                                 #
        # ------------------------------------------------------------------ #
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token,       std=0.02)
        nn.init.trunc_normal_(self.pos_embed,        std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.head.weight,      std=0.02)
        nn.init.zeros_(self.head.bias)

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        logits   : (B, num_classes)
        features : (B, d_model)   — [CLS] embedding (for contrastive loss)
        """
        B = x.size(0)

        # Patch embedding: (B, 3, 224, 224) → (B, d_model, 14, 14) → (B, 196, d_model)
        x = self.patch_embed(x)                    # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)           # (B, N, d_model)

        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)            # (B, N+1, d_model)
        x   = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)                         # (B, N+1, d_model)
        x = self.norm(x)

        # [CLS] output → classification
        cls_out  = x[:, 0, :]                       # (B, d_model)
        features = cls_out
        logits   = self.head(self.dropout(cls_out)) # (B, num_classes)

        return logits, features

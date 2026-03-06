"""
model_resnet18.py

ResNet18-backbone token model for melting curve classification.

Architecture
------------
image (B, 3, 224, 224)
  → ResNet18 up to layer4  (frozen, pretrained ImageNet)
  → feature map  (B, 512, 7, 7)
  → flatten spatial  → (B, 49, 512)
  → linear projection  → (B, 49, d_model)
  → prepend [CLS] token  → (B, 50, d_model)
  → learnable positional embedding
  → 1-layer MH self-attention encoder (BERT-style, Pre-LN)
  → [CLS] output  (B, d_model)
  → dropout → Linear  → logits  (B, num_classes)

Why layer4?
-----------
layer4 is the last residual block before global-avg-pool.  Its 7×7 spatial
grid gives 49 highly semantic tokens (512-dim each), analogous to the 217
patch tokens in the original model.py.  Higher layers (layer3: 196 tokens)
are an easy alternative — just swap BACKBONE_LAYER and BACKBONE_DIM below.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18TokenModel(nn.Module):
    """
    Frozen ResNet18 backbone → spatial feature tokens → [CLS] + 1-layer
    MH self-attention (BERT-style classification head).

    Parameters
    ----------
    num_classes   : int   — number of output classes
    d_model       : int   — transformer hidden dim (default 128)
    n_heads       : int   — number of attention heads (default 8)
    d_ff          : int   — feedforward dim inside transformer (default 256)
    dropout       : float — dropout rate (default 0.2)
    freeze_backbone : bool — freeze ResNet18 weights (default True)
    """

    # ResNet18 layer4 output: (B, 512, 7, 7)
    BACKBONE_DIM  = 512
    NUM_SPATIAL   = 7 * 7   # = 49 tokens

    def __init__(
        self,
        num_classes:      int,
        d_model:          int   = 128,
        n_heads:          int   = 8,
        d_ff:             int   = 256,
        dropout:          float = 0.2,
        freeze_backbone:  bool  = True,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Backbone: ResNet18 up to (and including) layer4                     #
        # ------------------------------------------------------------------ #
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------ #
        # Token projection: BACKBONE_DIM (512) → d_model                     #
        # ------------------------------------------------------------------ #
        self.proj = nn.Linear(self.BACKBONE_DIM, d_model)

        # ------------------------------------------------------------------ #
        # [CLS] token + learnable positional embedding                        #
        # ------------------------------------------------------------------ #
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed  = nn.Parameter(
            torch.randn(1, self.NUM_SPATIAL + 1, d_model) * 0.02
        )

        # ------------------------------------------------------------------ #
        # 1-layer transformer encoder (BERT-style, Pre-LayerNorm)             #
        # ------------------------------------------------------------------ #
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,                    # Pre-LN: more stable
        )
        self.norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------ #
        # Classification head                                                 #
        # ------------------------------------------------------------------ #
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
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
        features : (B, d_model)   — [CLS] embedding before head (for contrastive loss)
        """
        B = x.size(0)

        # --- backbone (frozen: no gradient) ---
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                feat = self.backbone(x)          # (B, 512, 7, 7)
        else:
            feat = self.backbone(x)

        # --- spatial tokens ---
        tokens = feat.flatten(2).transpose(1, 2)   # (B, 49, 512)
        tokens = self.proj(tokens)                  # (B, 49, d_model)

        # --- prepend [CLS] ---
        cls    = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, 50, d_model)
        tokens = tokens + self.pos_embed             # + positional embedding

        # --- 1-layer self-attention ---
        tokens = self.encoder(tokens)                # (B, 50, d_model)
        tokens = self.norm(tokens)

        # --- [CLS] output → classification ---
        cls_out  = tokens[:, 0, :]                   # (B, d_model)
        features = cls_out
        logits   = self.head(self.dropout(cls_out))  # (B, num_classes)

        return logits, features

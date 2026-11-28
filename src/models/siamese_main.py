import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Cosine head initialisation. cos lies in [0, 1] here because the features are
# post-ReLU (non-negative), so |logit| <= MAX_SCALE * (1 + |bias|).
INIT_SCALE = 10.0
INIT_BIAS = 0.5
MAX_SCALE = 100.0  # upper clamp on the learned scale (as in CLIP's logit_scale)


class AttentionalSiameseNetwork(nn.Module):
    def __init__(self, input_shape=(224, 224), dropout_prob=0.0, freeze_backbone=False):
        super(AttentionalSiameseNetwork, self).__init__()
        
        # --- שלב 1: רשת הקשב (ResNet18) ---
        # אנו משתמשים ב-ResNet18 כדי לייצר מפות תכונות מרחביות
        # weights=models.ResNet18_Weights.DEFAULT equivalent to pretrained=True
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # אנו מסירים את ה-AvgPool וה-FC הסופיים כדי לשמור על המימדים המרחביים (Spatial Dimensions)
        # הפלט של layer4 הוא בדרך כלל [Batch, 512, 7, 7] עבור תמונה של 224x224
        self.attention_backbone = nn.Sequential(*list(resnet18.children())[:-2])
        
        # שכבת עזר קטנה לכיווץ הערוצים של מפת הקשב (מ-512 ל-1)
        self.attn_conv = nn.Conv2d(512, 1, kernel_size=1) 

        # --- שלב 2: רשת התכונות העמוקות (ResNet34 - Balanced) ---
        # User requested "ResNet32" (ResNet34 is the standard pytorch equivalent)
        # This is deeper than 18 but faster than 50.
        resnet_features = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # ResNet34 output is [Batch, 512, 7, 7] (same channels as 18, but deeper features)
        self.feature_backbone = nn.Sequential(*list(resnet_features.children())[:-2])
        
        # --- שלב 3: Cosine scoring head ---
        # logit = scale * (cos - bias). Replaces Linear(1,1) on a raw sum of
        # 25,088 squared differences, whose range (10,778-62,837 at init) pinned
        # sigmoid at exactly 1.0 for every pair and overflowed fp16.
        # scale is kept positive (exp of a log), so higher cosine always means
        # "more likely same"; the old Linear weight had to learn its own sign.
        self.log_scale = nn.Parameter(torch.tensor(math.log(INIT_SCALE)))
        self.bias = nn.Parameter(torch.tensor(INIT_BIAS))

        # Dropout Layer
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # הקפאת משקולות (אופציונלי - תלוי באסטרטגיית האימון שלך)
        if freeze_backbone:
             self._freeze_weights(self.attention_backbone)
             self._freeze_weights(self.feature_backbone)
             
        # Ensure these are always trainable
        for param in self.attn_conv.parameters():
            param.requires_grad = True
        self.log_scale.requires_grad = True
        self.bias.requires_grad = True
        
    def _freeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the backbone for fine-tuning."""
        print("Unfreezing backbones for fine-tuning...")
        for param in self.attention_backbone.parameters():
            param.requires_grad = True
        for param in self.feature_backbone.parameters():
            param.requires_grad = True

    def forward_attention(self, img1, img2):
        """
        מייצר את מפת הקשב המשותפת על בסיס הדמיון בין התמונות
        """
        # 1. חילוץ תכונות מרחביות מ-ResNet18
        # Output shape: [Batch, 512, 7, 7]
        f1_coarse = self.attention_backbone(img1)
        f2_coarse = self.attention_backbone(img2)
        
        # 2. חישוב דמיון (Co-Attention)
        # ישנן דרכים רבות לחשב זאת. כאן נשתמש בגישה של הכפלת אלמנטים + קונבולוציה ללמידת הדמיון
        # אנו בודקים איפה הפיקסלים "מסכימים"
        similarity_map = f1_coarse * f2_coarse # Element-wise multiplication
        
        # 3. הפיכה למפת קשב בערוץ בודד [Batch, 1, 7, 7]
        attn_map = self.attn_conv(similarity_map)
        
        # 4. נרמול לטווח [0, 1] באמצעות Sigmoid
        # פיקסלים חשובים יקבלו ~1, רקע יקבל ~0
        attn_map = torch.sigmoid(attn_map)
        
        return attn_map

    def forward(self, img1, img2):
        # --- שלב A: יצירת מפת הקשב (ResNet18) ---
        # המפה אומרת לנו: "איפה שתי התמונות דומות?"
        # Shape: [Batch, 1, 7, 7]
        attention_mask = self.forward_attention(img1, img2)
        
        # --- שלב B: חילוץ תכונות עמוק (ResNet34) ---
        # Output shape: [Batch, 512, 7, 7]
        feat1_deep = self.feature_backbone(img1)
        feat2_deep = self.feature_backbone(img2)
        
        # Apply Dropout to feature maps
        feat1_deep = self.dropout(feat1_deep)
        feat2_deep = self.dropout(feat2_deep)
        
        logits = self.score(feat1_deep, feat2_deep, attention_mask)

        # Return Logits directly for BCEWithLogitsLoss
        return logits, attention_mask

    def score(self, feat1, feat2, attention_mask):
        """[B,C,H,W] feature maps + [B,1,H,W] mask -> [B,1] same-person logits.

        --- Soft Gating (הליבה של המודל) ---
        Each map is multiplied by the shared attention mask, flattened, and the
        two are compared with cosine similarity. Locations still correspond one
        to one (the same cell of both photos is compared), and a cell with mask
        ~0 contributes ~nothing, as before. Cosine is invariant to the overall
        size of the features and of the mask, so the logit cannot grow with
        feature magnitude: |logit| <= MAX_SCALE * (1 + |bias|).
        """
        g1 = (feat1 * attention_mask).flatten(1)
        g2 = (feat2 * attention_mask).flatten(1)
        cos = F.cosine_similarity(g1, g2, dim=1, eps=1e-8)  # [B]; 0 if a map is all-zero
        scale = self.log_scale.exp().clamp(max=MAX_SCALE)
        return (scale * (cos - self.bias)).view(-1, 1)
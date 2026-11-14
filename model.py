
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HazardAwareAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        hidden = max(channels // 4, 32)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_map = self.channel_attention(x)
        x = x * channel_map

        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)

        spatial_map = self.spatial_attention(
            torch.cat([avg_map, max_map], dim=1)
        )

        return x * spatial_map


class MHViTModel(nn.Module):

    def __init__(
        self,
        num_classes=4,
        use_multiscale=True,
        use_hazard_attention=True,
        use_fusion=True,
        use_confidence_head=True,
        pretrained=False
    ):
        super().__init__()

        self.vit = models.vit_b_16(weights=None)
        self.embed_dim = self.vit.hidden_dim

        self.use_multiscale = use_multiscale
        self.use_hazard_attention = use_hazard_attention
        self.use_fusion = use_fusion
        self.use_confidence_head = use_confidence_head

        self.capture_layers = [4, 8, 12]

        self.level_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim)
            for _ in self.capture_layers
        ])

        self.level_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, 256, 1),
                nn.BatchNorm2d(256),
                nn.GELU()
            )
            for _ in range(3)
        ])

        self.hazard_attention = HazardAwareAttention(
            self.embed_dim
        )

        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, 256, 1),
                nn.BatchNorm2d(256),
                nn.GELU()
            )
            for _ in range(3)
        ])

        base_dim = 256 * 3 + self.embed_dim

        fusion_input_dim = (
            base_dim + (256 * 3 if use_multiscale else 0)
        )

        if use_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Dropout(0.10)
            )
            final_dim = self.embed_dim
        else:
            self.fusion = nn.Identity()
            final_dim = fusion_input_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(final_dim, max(final_dim // 2, 256)),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(max(final_dim // 2, 256), num_classes)
        )

        if use_confidence_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(final_dim, 256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, 1)
            )
        else:
            self.confidence_head = None

    def encode_intermediate(self, x):

        x = self.vit._process_input(x)

        cls_token = self.vit.class_token.expand(
            x.shape[0], -1, -1
        )

        x = torch.cat([cls_token, x], dim=1)

        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        features = []

        for layer_idx, block in enumerate(
            self.vit.encoder.layers, start=1
        ):
            x = block(x)

            if layer_idx in self.capture_layers:
                features.append(x)

        final_tokens = self.vit.encoder.ln(x)
        cls_token = final_tokens[:, 0, :]

        normalized_features = [
            norm(feat)
            for feat, norm in zip(
                features, self.level_norms
            )
        ]

        return normalized_features, cls_token

    def forward(self, x):

        level_features, cls_token = self.encode_intermediate(x)

        level_vectors = []

        for feat, projection in zip(
            level_features, self.level_projections
        ):

            patches = feat[:, 1:, :]
            b, n, d = patches.shape
            side = int(math.sqrt(n))

            spatial = patches.transpose(1, 2).reshape(
                b, d, side, side
            )

            spatial = projection(spatial)

            pooled = F.adaptive_avg_pool2d(
                spatial, 1
            ).flatten(1)

            level_vectors.append(pooled)

        deepest_patches = level_features[-1][:, 1:, :]

        b, n, d = deepest_patches.shape
        side = int(math.sqrt(n))

        deepest_spatial = deepest_patches.transpose(
            1, 2
        ).reshape(b, d, side, side)

        if self.use_hazard_attention:
            attended = self.hazard_attention(deepest_spatial)
        else:
            attended = deepest_spatial

        vectors = level_vectors

        if self.use_multiscale:

            scale_vectors = []

            for pool_size, projection in zip(
                [7, 4, 1], self.scale_projections
            ):

                pooled = F.adaptive_avg_pool2d(
                    attended,
                    (pool_size, pool_size)
                )

                pooled = projection(pooled)

                pooled = F.adaptive_avg_pool2d(
                    pooled, 1
                ).flatten(1)

                scale_vectors.append(pooled)

            vectors = vectors + scale_vectors

        fused_input = torch.cat(
            vectors + [cls_token], dim=1
        )

        final_repr = self.fusion(fused_input)

        logits = self.classifier(final_repr)

        if self.confidence_head is not None:
            confidence = self.confidence_head(
                final_repr
            ).squeeze(1)
        else:
            confidence = None

        return logits, confidence

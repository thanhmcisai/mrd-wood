"""
Wood Spatial — Backbone Extractors
====================================
GlobalExtractor  — frozen backbone, returns L2-normalized pooled features (N, D)
SpatialExtractor — frozen backbone, returns intermediate spatial feature maps (B, C, H, W)
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from wood_spatial.config import BACKBONE_CONFIGS

logger = logging.getLogger(__name__)


class GlobalExtractor(nn.Module):
    """
    Frozen backbone feature extractor with L2-normalized output.

    Removes classifier head, applies global average pooling if needed.
    Adapted from github_package_v2/wood_benchmark/core/backbone.py.
    """

    def __init__(self, backbone_id: str):
        super().__init__()
        cfg = BACKBONE_CONFIGS[backbone_id]
        self.backbone_id = backbone_id
        model_id = cfg['model_id']
        img_size = cfg['img_size']
        extra_kwargs = dict(cfg.get('extra', {}))

        use_global_pool = not cfg.get('no_global_pool', False)
        pool_kwargs = {'global_pool': 'avg'} if use_global_pool else {}

        create_kwargs = dict(extra_kwargs)
        create_kwargs['img_size'] = img_size
        try:
            self.backbone = timm.create_model(
                model_id, pretrained=True, num_classes=0,
                **pool_kwargs, **create_kwargs,
            )
        except TypeError:
            create_kwargs.pop('img_size', None)
            try:
                self.backbone = timm.create_model(
                    model_id, pretrained=True, num_classes=0,
                    **pool_kwargs, **create_kwargs,
                )
            except TypeError:
                self.backbone = timm.create_model(
                    model_id, pretrained=True, num_classes=0, **pool_kwargs,
                )

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        logger.info('Loaded GlobalExtractor: %s (feat_dim=%d)',
                     backbone_id, self.feat_dim)

    @property
    def feat_dim(self) -> int:
        """Output feature dimensionality."""
        if hasattr(self.backbone, 'num_features'):
            return self.backbone.num_features
        return self.backbone.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone(x)
            if feat.dim() > 2:
                feat = feat.mean(dim=list(range(2, feat.dim())))
        return F.normalize(feat, dim=-1)


class SpatialExtractor(nn.Module):
    """
    Frozen backbone that returns intermediate spatial feature maps.

    Uses timm's features_only=True with out_indices=(2,) for CNN/HRNet/Swin.
    For DINOv2 (ViT), uses get_intermediate_layers to extract spatial tokens.

    Returns feature map of shape (B, C, H_f, W_f) at stride ~16 resolution.
    """

    def __init__(self, backbone_id: str):
        super().__init__()
        cfg = BACKBONE_CONFIGS[backbone_id]
        self.backbone_id = backbone_id
        self.is_dinov2 = 'dinov2' in cfg['model_id']
        model_id = cfg['model_id']
        img_size = cfg['img_size']
        extra_kwargs = dict(cfg.get('extra', {}))

        if self.is_dinov2:
            # DINOv2 ViT: load full model, use get_intermediate_layers
            create_kwargs = dict(extra_kwargs)
            create_kwargs['img_size'] = img_size
            try:
                self.backbone = timm.create_model(
                    model_id, pretrained=True, num_classes=0,
                    **create_kwargs,
                )
            except TypeError:
                create_kwargs.pop('img_size', None)
                self.backbone = timm.create_model(
                    model_id, pretrained=True, num_classes=0,
                    **create_kwargs,
                )
            self.patch_size = self.backbone.patch_embed.patch_size[0]
        else:
            # CNN/HRNet/Swin: use features_only
            create_kwargs = dict(extra_kwargs)
            create_kwargs['img_size'] = img_size
            try:
                self.backbone = timm.create_model(
                    model_id, pretrained=True,
                    features_only=True, out_indices=(2,),
                    **create_kwargs,
                )
            except TypeError:
                create_kwargs.pop('img_size', None)
                try:
                    self.backbone = timm.create_model(
                        model_id, pretrained=True,
                        features_only=True, out_indices=(2,),
                        **create_kwargs,
                    )
                except TypeError:
                    self.backbone = timm.create_model(
                        model_id, pretrained=True,
                        features_only=True, out_indices=(2,),
                    )

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        logger.info('Loaded SpatialExtractor: %s (is_dinov2=%s)',
                     backbone_id, self.is_dinov2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature map.

        Returns
        -------
        torch.Tensor of shape (B, C, H_f, W_f)
        """
        with torch.no_grad():
            if self.is_dinov2:
                return self._forward_dinov2(x)
            else:
                features = self.backbone(x)
                feat = features[0]  # out_indices=(2,) returns a list of 1
                # Some timm backbones (notably Swin) return NHWC feature maps.
                # The downstream spatial pipeline expects NCHW.
                if feat.dim() == 4 and feat.shape[-1] > feat.shape[1]:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                return feat

    def _forward_dinov2(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from DINOv2 ViT intermediate layer."""
        B, _, H, W = x.shape
        H_f = H // self.patch_size
        W_f = W // self.patch_size

        # Get intermediate layer output (block index 6 of 12)
        # The output is (B, N_patches, C) without CLS token when using this API
        intermediates = self.backbone.get_intermediate_layers(x, n=[6])
        tokens = intermediates[0]  # (B, N, C)

        # Handle CLS token if present
        if tokens.shape[1] == H_f * W_f + 1:
            tokens = tokens[:, 1:, :]  # remove CLS
        elif tokens.shape[1] != H_f * W_f:
            # Fallback: try without CLS token adjustment
            expected = H_f * W_f
            tokens = tokens[:, -expected:, :]

        C = tokens.shape[-1]
        spatial = tokens.reshape(B, H_f, W_f, C).permute(0, 3, 1, 2)
        return spatial


class GradCAMExtractor(nn.Module):
    """
    Backbone with a temporary linear classifier for Grad-CAM computation.

    Hooks the last convolutional/block output and computes standard Grad-CAM.
    Also supports gradient-free Centroid-CAM as a simpler alternative.
    """

    def __init__(self, backbone_id: str, n_classes: int):
        super().__init__()
        cfg = BACKBONE_CONFIGS[backbone_id]
        self.backbone_id = backbone_id
        model_id = cfg['model_id']
        img_size = cfg['img_size']
        extra_kwargs = dict(cfg.get('extra', {}))
        self.is_dinov2 = 'dinov2' in model_id

        # Load full model with classifier
        create_kwargs = dict(extra_kwargs)
        create_kwargs['img_size'] = img_size
        try:
            self.backbone = timm.create_model(
                model_id, pretrained=True, num_classes=0,
                **create_kwargs,
            )
        except TypeError:
            create_kwargs.pop('img_size', None)
            try:
                self.backbone = timm.create_model(
                    model_id, pretrained=True, num_classes=0,
                    **create_kwargs,
                )
            except TypeError:
                self.backbone = timm.create_model(
                    model_id, pretrained=True, num_classes=0,
                )

        feat_dim = self.backbone.num_features if hasattr(self.backbone, 'num_features') else self.backbone.embed_dim
        self.classifier = nn.Linear(feat_dim, n_classes)

        # Freeze backbone, only classifier is trainable
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Hook storage
        self._activation = None
        self._gradient = None

    def init_classifier_from_centroids(self, centroids: torch.Tensor):
        """
        Initialize classifier weights from class centroids.

        centroids: (n_classes, feat_dim), L2-normalized
        """
        with torch.no_grad():
            self.classifier.weight.copy_(centroids)
            self.classifier.bias.zero_()

    def _hook_activation(self, module, input, output):
        self._activation = output

    def _hook_gradient(self, module, grad_in, grad_out):
        self._gradient = grad_out[0]

    def compute_gradcam(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap for a single image.

        Returns (H_f, W_f) heatmap in [0, 1].
        """
        self.eval()
        self.classifier.requires_grad_(True)

        # Register hooks on the last feature layer
        # For timm models, forward_features returns pre-pool features
        feat = self.backbone.forward_features(x)

        if feat.dim() == 3:
            # ViT/DINOv2: (B, N, C) -> need spatial reshape
            B = feat.shape[0]
            # Remove CLS token if present
            if hasattr(self.backbone, 'patch_embed'):
                ps = self.backbone.patch_embed.patch_size[0]
                H_f = x.shape[2] // ps
                W_f = x.shape[3] // ps
                if feat.shape[1] == H_f * W_f + 1:
                    feat_spatial = feat[:, 1:, :]
                else:
                    feat_spatial = feat[:, -H_f * W_f:, :]
                feat_spatial = feat_spatial.reshape(B, H_f, W_f, -1).permute(0, 3, 1, 2)
            else:
                feat_spatial = feat
        elif feat.dim() == 4:
            feat_spatial = feat  # CNN: already (B, C, H, W)
        else:
            raise ValueError(f'Unexpected feature dim: {feat.dim()}')

        # Enable gradient for the spatial features
        feat_spatial.requires_grad_(True)

        # Pool and classify
        pooled = feat_spatial.mean(dim=[2, 3])  # (B, C)
        pooled = F.normalize(pooled, dim=-1)
        logits = self.classifier(pooled)

        # Backward
        self.zero_grad()
        logits[0, target_class].backward()

        # Grad-CAM
        grads = feat_spatial.grad  # (B, C, H, W)
        weights = grads.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        cam = (weights * feat_spatial).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam).squeeze(0).squeeze(0)  # (H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach()

"""
src/multimodal_model.py -- Multimodal encoders and fusion model.

Implements §4.2 of the Methodology (Multimodal Encoding and Fusion):
  - EHREncoder:            MLP that maps tabular features → ℝ^EMBED_DIM
  - VitalsEncoder:         1D-CNN that maps (T, C) time-series → ℝ^EMBED_DIM
  - ImagingEncoder:        2D-CNN that maps (1, H, W) images → ℝ^EMBED_DIM
  - MultimodalFusionModel: Fuses encoder outputs with presence mask → classifier

Missing-modality handling:
  For any absent modality k, we set z_k = 0 ∈ ℝ^EMBED_DIM and m_k = 0.
  The mask vector m is concatenated into the fusion input so the downstream
  classifier knows which embeddings are real and which are zero-fill.
"""
import torch
import torch.nn as nn

import src.config as config


class EHREncoder(nn.Module):
    """MLP encoder for tabular EHR features → EMBED_DIM-d embedding."""

    def __init__(
        self,
        n_features: int,
        embed_dim: int = config.EMBED_DIM,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features) → (B, embed_dim)"""
        return self.net(x)


class VitalsEncoder(nn.Module):
    """1D-CNN encoder for time-series vitals → EMBED_DIM-d embedding.

    Architecture: three Conv1d blocks with batch-norm and max-pool,
    followed by adaptive global average pooling → linear → LayerNorm.
    """

    def __init__(
        self,
        in_channels: int = config.VITALS_CHANNELS,
        embed_dim: int = config.EMBED_DIM,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pooling → (B, 128, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, embed_dim)"""
        # Conv1d expects (B, C, T), so transpose
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze(-1)  # (B, 128)
        return self.fc(x)


class ImagingEncoder(nn.Module):
    """Lightweight 2D-CNN encoder for medical images → EMBED_DIM-d embedding.

    Architecture: four Conv2d blocks with batch-norm and max-pool / adaptive
    pooling, followed by linear → LayerNorm.
    """

    def __init__(
        self,
        in_channels: int = config.IMG_CHANNELS,
        embed_dim: int = config.EMBED_DIM,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: H×W → H/2 × W/2
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: → H/4 × W/4
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: → H/8 × W/8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4: → 1×1 (global avg pool)
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, embed_dim)"""
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # (B, 64)
        return self.fc(x)


class ImageClassifier(nn.Module):
    """Binary image-recognition model for image-only datasets.

    This is deliberately separate from the fusion model: an image-only sample
    should use the CNN directly, rather than passing fabricated EHR/vitals
    tensors through a multimodal model.
    """

    def __init__(self, embed_dim: int = config.EMBED_DIM, dropout: float = config.DROPOUT):
        super().__init__()
        self.encoder = ImagingEncoder(config.IMG_CHANNELS, embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x_img)).squeeze(-1)


class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion model with missing-modality handling.

    Architecture (per §4.2 of the Methodology):
      1. Three encoders produce z_ehr, z_vitals, z_img ∈ ℝ^EMBED_DIM
      2. Missing modalities: z_k *= 0 (via mask), m_k = 0
      3. Concatenate: [z_ehr·m_ehr ‖ z_vit·m_vit ‖ z_img·m_img ‖ m]
      4. LayerNorm → Linear projection → FUSED_DIM-d patient representation
      5. Classifier head → 1 logit (use BCEWithLogitsLoss externally)
    """

    def __init__(
        self,
        n_ehr_features: int,
        embed_dim: int = config.EMBED_DIM,
        fused_dim: int = config.FUSED_DIM,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_modalities = 3

        # Modality-specific encoders
        self.ehr_encoder = EHREncoder(n_ehr_features, embed_dim, dropout)
        self.vitals_encoder = VitalsEncoder(config.VITALS_CHANNELS, embed_dim)
        self.imaging_encoder = ImagingEncoder(config.IMG_CHANNELS, embed_dim)

        # Fusion layer: 3 * embed_dim + 3 (mask bits)
        fusion_input_dim = self.n_modalities * embed_dim + self.n_modalities
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x_ehr: torch.Tensor,
        x_vitals: torch.Tensor,
        x_img: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_ehr         : (B, n_ehr_features)  — tabular features
        x_vitals      : (B, T, C)            — time-series vitals
        x_img         : (B, 1, H, W)         — medical image
        modality_mask : (B, 3)               — [m_ehr, m_vit, m_img] ∈ {0, 1}

        Returns
        -------
        logits : (B,) — raw logits for binary classification
        """
        # Encode each modality
        z_ehr = self.ehr_encoder(x_ehr)        # (B, embed_dim)
        z_vit = self.vitals_encoder(x_vitals)  # (B, embed_dim)
        z_img = self.imaging_encoder(x_img)    # (B, embed_dim)

        # Apply presence mask: zero out embeddings for missing modalities
        m_ehr = modality_mask[:, 0:1]  # (B, 1)
        m_vit = modality_mask[:, 1:2]  # (B, 1)
        m_img = modality_mask[:, 2:3]  # (B, 1)

        z_ehr = z_ehr * m_ehr  # zero-fill if m_ehr = 0
        z_vit = z_vit * m_vit
        z_img = z_img * m_img

        # Concatenate embeddings + mask → (B, 3*embed_dim + 3)
        fused_input = torch.cat([z_ehr, z_vit, z_img, modality_mask], dim=1)

        # Fusion projection → (B, fused_dim)
        z_patient = self.fusion(fused_input)

        # Classify → (B, 1) → squeeze → (B,)
        logits = self.classifier(z_patient)
        return logits.squeeze(-1)

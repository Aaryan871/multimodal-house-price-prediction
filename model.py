import torch
import torch.nn as nn
from torchvision import models


class CNNBranch(nn.Module):
    def __init__(self, out_dim=256, freeze_backbone=False):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        in_features = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.project = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.project(features)


class TabularBranch(nn.Module):

    def __init__(self, in_dim=3, out_dim=64):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(in_dim, 128),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.3),

            nn.Linear(128, out_dim),

            nn.ReLU()

        )

    def forward(self, x):
        return self.net(x)


class MultimodalHousePriceModel(nn.Module):

    def __init__(
        self,
        tabular_in_dim=3,
        img_feat_dim=256,
        tab_feat_dim=64,
        freeze_backbone=False
    ):

        super().__init__()

        self.cnn_branch = CNNBranch(
            out_dim=img_feat_dim,
            freeze_backbone=freeze_backbone
        )

        self.tabular_branch = TabularBranch(
            tabular_in_dim,
            out_dim=tab_feat_dim
        )

        concat_dim = img_feat_dim + tab_feat_dim

        self.regression_head = nn.Sequential(

            nn.Linear(concat_dim, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 32),

            nn.ReLU(),

            nn.Linear(32, 1)

        )

    def forward(self, image, tabular):

        img_features = self.cnn_branch(image)

        tab_features = self.tabular_branch(tabular)

        combined = torch.cat(
            [img_features, tab_features],
            dim=1
        )

        output = self.regression_head(combined)

        return output.squeeze(1)

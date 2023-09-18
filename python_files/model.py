from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


extractor_model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
extractor_model.classifier[3]=nn.Identity()


class Mobilenet_reg(nn.Module):
    def __init__(self,mobilenet_model):
        super().__init__()
        self.mobilenet_model=mobilenet_model
        self.classifier=nn.Linear(1024,31)
        self.regression=nn.Sequential(
            nn.Linear(1024,512),
            nn.GELU(),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        extraction=self.mobilenet_model(x)
        class_pred=self.classifier(extraction)
        regression_pred=self.regression(extraction)

        return class_pred,regression_pred

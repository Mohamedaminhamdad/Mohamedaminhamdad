import os
from easydict import EasyDict as edict
import torch
import torch.nn as nn

class Basicnet(nn.Module):
    def __init__(self, backbone, segmentation):
        super(Basicnet, self).__init__()
        self.backbone = backbone
        self.segmentation = segmentation


    def forward(self, x):                     # 
        features = self.backbone(x)           # 
        segmentation = self.segmentation(features) # 
        return segmentation

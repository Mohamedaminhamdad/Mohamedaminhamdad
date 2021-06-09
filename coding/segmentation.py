import torch.nn as nn
import math
#import ref
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck

class Segmentationm(nn.Module):
    def __init__(self, in_channels, num_layers=2, kernel_size=3, output_kernel_size=1,
                 output_dim=1, freeze=False, with_bias_end=True):
        self.freeze = freeze
        super(Segmentationm, self).__init__()
        padding=1
        output_padding=0
        features = nn.ModuleList()
        for i in range(num_layers):
            if i==0: 
                _in_channels=in_channels
            else: 
                _in_channels=num_filters
            num_filters=int(_in_channels/2)
            features.append(nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,output_padding=output_padding, bias=False))
            features.append(nn.BatchNorm2d(num_filters))
            features.append(nn.ReLU(inplace=True))
            features.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=2, bias=False))
            features.append(nn.BatchNorm2d(num_filters))
            features.append(nn.ReLU(inplace=True))
            features.append(nn.Conv2d(num_filters, num_filters, kernel_size=2, stride=1, padding=0, bias=False))
            features.append(nn.BatchNorm2d(num_filters))
            features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(num_filters, output_dim, kernel_size=output_kernel_size, padding=0, bias=True))
        features.append(nn.BatchNorm2d(output_dim))
        features.append(nn.ReLU(inplace=True))
        self.features=features
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end and (m.bias is not None):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                return x.detach()
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            return x

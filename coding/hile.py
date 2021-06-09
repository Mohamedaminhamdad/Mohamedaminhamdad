from resnet_backbone import ResNetBackboneNet
import cv2 
import numpy
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import CustomImageDataset
from modeli import Basicnet
from segmentation import Segmentationm
from trainingseg import training_seg
from torch.utils.tensorboard import SummaryWriter
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
block_type, layers, channels, name = resnet_spec[18]

def _worker_init_fn():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)

writer = SummaryWriter()
net_backbone=ResNetBackboneNet(block_type, layers, in_channel=3, freeze=False)
net_segmentation=Segmentationm(channels[-1], num_layers=4, kernel_size=3, output_kernel_size=1,output_dim=1, freeze=False, with_bias_end=True)
model=Basicnet(net_backbone,net_segmentation)
train_dataloader=DataLoader(CustomImageDataset(),batch_size=12,shuffle=True,num_workers=int(12), worker_init_fn=_worker_init_fn())
train_features, train_labels = next(iter(train_dataloader))
params_lr_list=[]
params_lr_list.append({'params': filter(lambda p: p.requires_grad, net_backbone.parameters()),
                                   'lr': float(1e-4)})
params_lr_list.append({'params': filter(lambda p: p.requires_grad, net_segmentation.parameters()),
                                   'lr': float(1e-4)})
optimizer = torch.optim.RMSprop(params_lr_list, alpha=0.99, eps=float(1e-8),
                                            weight_decay=0.0, momentum=0.0)
criterion = nn.BCEWithLogitsLoss()
criterion=criterion.cuda('1')
optimizer=optimizer.cuda('1')
model=model.cuda('1')                                 
training_seg(model,optimizer,train_dataloader,criterion)
torch.save(model.state_dict(), './')

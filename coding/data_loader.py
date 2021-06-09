import os 
from glob import glob
import pandas as pandas
import yaml
import cv2
import numpy as np
class CustomImageDataset():
    def __init__(self, transform=None, target_transform=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.img = glob(os.path.join(dir_path,'ape','*.png'))
        self.img_labels=glob(os.path.join(dir_path,'mask', '*.png'))
        self.transform=None
        self.target_transform=None
    def __len__(self):
        length = len(self.img_labels)
        return length
    def __getitem__(self, idx):
        img_=cv2.imread(self.img[idx])
        #img_=img_.reshape(240,320,12)
        img_labels_=cv2.imread(self.img_labels[idx],cv2.IMREAD_GRAYSCALE)
        img_labels_=img_labels_/255.
        rgb = img_.transpose(2, 0, 1).astype(np.float32) / 255.
        #img_labels_=img_labels_.transpose(2, 0, 1).astype(np.float32) / 255.
        return rgb, img_labels_
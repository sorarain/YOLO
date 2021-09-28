import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

class Augmentation(object):
    def __init__(self,size,mean,std):
        self.size = size
        self.mean = mean;
        self.std = std;
    def __call__(self, image,box,label):
        img = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        img /= 255;
        img -= self.mean
        img /= self.std
        return img,box,label;


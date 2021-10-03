import os
import xml.etree.ElementTree as ET

import numpy
import numpy as np
import torch
import torch.utils.data as data
import cv2
from augmentation import Augmentation
import torch.nn as nn
import torchvision.models as torchmodel

def cal_IOU(bbox1,bbox2):
    x11,y11,x12,y12 = bbox1
    x21,y21,x22,y22 = bbox2
    if x12 <= x11 or x22 <= x21 or y12 <= y11 or y22 <= y21:
        return 0;
    bbox_intersect = [max(x11,x21),
                      max(y11,y21),
                      min(x12,x22),
                      min(y12,y22)]
    bbox1_S = (x12 - x11) * (y12 - y11)
    bbox2_S = (x22 - x21) * (y22 - y21)
    bbox_intersect_S = (bbox_intersect[2] - bbox_intersect[0]) \
                       * (bbox_intersect[3] - bbox_intersect[1])
    IOU = bbox_intersect_S / (1e-6 + bbox1_S + bbox2_S)
    return IOU

if __name__ == "__main__":
    bbox1 = [0.25,0.25,0.75,0.75]
    bbox2 = [0.5,0.5,1,1]
    print(cal_IOU(bbox1,bbox2))
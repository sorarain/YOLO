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
from tools import cal_IOU

class YOLOnet(nn.Module):
    def __init__(self):
        super(YOLOnet,self).__init__()
        ##use YOLOv1 backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(64,192,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(192,128,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128,256,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,256,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,512,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(512,256,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,512,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512,256,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,512,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512,256,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,512,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512,256,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256,512,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512,512,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512,1024,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(1024,512,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512,1024,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(1024,512,1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512,1024,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(1024,1024,3,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(1024,1024,3,stride=2,padding=1),
            nn.LeakyReLU(inplace=True),

        )
        self.fc_bone = nn.Sequential(
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(7 * 7 * 1024,4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096,7 * 7 * 30),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1,(30,7,7))
        )
        self.fc_pretrain = nn.Sequential(
            nn.AvgPool2d(7,stride=1),
            nn.Dropout(0.4),
            nn.Flatten(1,-1),
            nn.Linear(1024,1000),
        )

        #use resnet as backbone
        resnet = torchmodel.resnet34(pretrained=True)
        self.resnet_out_channels = resnet.fc.in_features
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(self.resnet_out_channels, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 7 * 7 * 30),
            nn.Sigmoid(),  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
            nn.Unflatten(1, (30,7, 7))
        )



    def forward(self,x):
        # x = self.backbone(x);
        # x = self.fc_bone(x);
        # x = self.fc_pretrain(x);
        x = self.resnet_backbone(x)
        x = self.Conv_layers(x);
        x = self.Conn_layers(x);
        return x

    def cal_loss(self,pred,labels):#pred [batch_size,30,B,B] 10=(x1,y1,w1,h1,P(object1),x2,y2,w2,h2,P(object2),20 P(class))
        self.pred = pred.double()
        labels = labels.double()#label[batch_size,30,B,B] 30 5*2+20 5(groundtruth 复制两份)
        grid_x,grid_y = 7,7
        coord_loss = 0;
        obj_loss = 0;
        noobj_loss = 0;
        class_loss = 0;
        batch_size = labels.size(0)
        for i in range(batch_size):
            for x in range(grid_x):
                for y in range(grid_y):
                    if labels[i,4,x,y] == 1:
                        bbox1_pred = [(self.pred[i,0,x,y] + x) / grid_x - self.pred[i,2,x,y] / 2.0,
                                      (self.pred[i,1,x,y] + x) / grid_x - self.pred[i,3,x,y] / 2.0,
                                      (self.pred[i,0,x,y] + x) / grid_x + self.pred[i,2,x,y] / 2.0,
                                      (self.pred[i,1,x,y] + x) / grid_x + self.pred[i,3,x,y] / 2.0,]

                        bbox2_pred = [(self.pred[i,5,x,y] + x) / grid_x - self.pred[i,7,x,y] / 2.0,
                                      (self.pred[i,6,x,y] + x) / grid_x - self.pred[i,8,x,y] / 2.0,
                                      (self.pred[i,5,x,y] + x) / grid_x + self.pred[i,7,x,y] / 2.0,
                                      (self.pred[i,6,x,y] + x) / grid_x + self.pred[i,8,x,y] / 2.0,]

                        bbox_gt = [labels[i,0,x,y] - labels[i,2,x,y] / 2.0,
                                   labels[i,1,x,y] - labels[i,3,x,y] / 2.0,
                                   labels[i, 0, x, y] + labels[i, 2, x, y] / 2.0,
                                   labels[i, 1, x, y] + labels[i, 3, x, y] / 2.0,]
                        IOU1 = cal_IOU(bbox1_pred,bbox_gt)
                        IOU2 = cal_IOU(bbox2_pred,bbox_gt)
                        if IOU1 >= IOU2:
                            coord_loss = coord_loss + 5 * torch.sum((self.pred[i,:2,x,y] - labels[i,:2,x,y])**2)
                            + 5 * torch.sum((self.pred[i,2:4,x,y].sqrt() - labels[i,2:4,x,y].sqrt())**2)
                            obj_loss = obj_loss + torch.sum((self.pred[i,4,x,y] - IOU1)**2)
                            noobj_loss = noobj_loss + 0.5 * torch.sum((self.pred[i,9,x,y] - IOU2)**2);
                        else:
                            coord_loss = coord_loss + 5 * torch.sum((self.pred[i, 5:7, x, y] - labels[i, 5:7, x, y]) ** 2)
                            + 5 * torch.sum((self.pred[i, 7:9, x, y].sqrt() - labels[i, 7:9, x, y].sqrt()) ** 2)
                            obj_loss = obj_loss + torch.sum((self.pred[i, 9, x, y] - IOU2) ** 2)
                            noobj_loss = noobj_loss + 0.5 *torch.sum((self.pred[i, 4, x, y] - IOU1) ** 2);
                        class_loss = class_loss + torch.sum((self.pred[i,10:,x,y] - labels[i,10:,x,y])**2)
                    else:
                        noobj_loss = noobj_loss + 0.5 * torch.sum(self.pred[i,[4,9],x,y]**2)
        loss = coord_loss + obj_loss + noobj_loss + class_loss
        return loss / batch_size






if __name__  == "__main__":
    img = torch.rand([1,3,448,448])
    net = YOLOnet();
    trans = net(img)
    tran = torch.zeros_like(trans)
    labels = torch.ones_like(trans)
    for i in range(trans.size(0)):
        for x in range(trans.size(2)):
            for y in range(trans.size(3)):
                tran[i,:5,x,y] = torch.Tensor([0.5,0.5,0.5,0.5,0.5])
                labels[i,:5,x,y] = torch.Tensor([0.75,0.75,0.5,0.5,1])
    # print(trans.shape)

    loss = net.cal_loss(tran,labels)

    print(loss)


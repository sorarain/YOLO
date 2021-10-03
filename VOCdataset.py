import os
import xml.etree.ElementTree as ET

import numpy
import numpy as np
import torch
import torch.utils.data as data
import cv2
from augmentation import Augmentation

voc_class = ['horse', 'aeroplane', 'dog', 'sofa', 'bottle', 'person', 'train', 'tvmonitor', 'chair',
             'diningtable', 'cow', 'bird', 'cat', 'bicycle', 'boat', 'motorbike', 'bus', 'car', 'pottedplant', 'sheep']

class dataVOC(data.Dataset):#D:\YOLO\redo\VOCdevkit\VOC2012
    def __init__(self,root = None,transforme = None):
        super(dataVOC, self).__init__()
        self.root = root#数据集地址
        self.transform_img = transforme#图片的transform
        self.trainveltxtpath = os.path.join(root,"ImageSets","Main","trainval.txt")#从VOC来开是用来目标检测训练和测试的所有数据
        self.xmlfilepath = os.path.join(root,"Annotations","%s.xml")#对应图片相关信息的xml地址
        self.jpegpath = os.path.join(root,"JPEGImages","%s.jpg")#对应图片地址
        self.xmllist = list();
        self.classtoindex = dict(zip(voc_class,range(len(voc_class))))#类别标签离散化
        for line in open(self.trainveltxtpath):
            self.xmllist.append([self.xmlfilepath,line.strip()])#数据集图片标签离散化

    def __len__(self):
        return len(self.xmllist)
    def transform_target(self,root,width,height):
        target = list()
        for ob in root.findall('object'):
            label = ob.find('name').text
            bndbox = ob.find('bndbox')
            xmin =  (int(bndbox.find('xmin').text) - 1);
            ymin =  (int(bndbox.find('ymin').text) - 1);
            xmax =  (int(bndbox.find('xmax').text) - 1);
            ymax =  (int(bndbox.find('ymax').text) - 1);#将boundingbox边框归一化
            x = (xmin + xmax) / 2.0;
            y = (ymin + ymax) / 2.0;
            w = (xmax - xmin)
            h = ymax - ymin;
            x /= 1.0 * height
            y /= 1.0 * width
            # x = 1.0 * x / 7.0 - int(x / 7.0)
            # y = 1.0 * y / 7.0 - int(y / 7.0)
            w /= 1.0 * height
            h /= 1.0 * width

        # target.append([xmin,ymin,xmax,ymax,self.classtoindex[label]])
            target.append([x, y, w, h, self.classtoindex[label]])
        return target;


    def get_item(self,index):
        xmlpath = os.path.join(self.xmlfilepath % self.xmllist[index][1])
        jpgpath = os.path.join(self.jpegpath % self.xmllist[index][1]);
        root = ET.parse(xmlpath).getroot();
        img = cv2.imread(jpgpath)
        height,width,channels = img.shape;
        target = self.transform_target(root,height,width)
        if self.transform_img is not None:
            target = np.array(target)
            img,box,label = self.transform_img(img,target[:,:4],target[:,4:]);#将图片和对应图片内的所有边框一起进行处理transform
            target = np.hstack((box,label))
        return torch.from_numpy(img),target#返回经过transform的图像和annotation信息


    def __getitem__(self, index):
        return self.get_item(index)
    def get_img(self,index):
        return cv2.imread(self.jpegpath % self.xmllist[index][1])#返回原图像
    def get_annotation(self,index):
        imgid = self.xmllist[index][1]
        root = ET.parse(self.xmlfilepath % imgid).getroot()#返回原annotation
        target = self.transform_target(root,1,1);
        return imgid,target

if __name__ == '__main__':
    size = 600;
    imgid = 7;
    basetransform = Augmentation(size = size,mean = (0,0,0),std=(1,1,1))
    vocdata = dataVOC(root = './VOCdevkit/VOC2012',transforme=basetransform)
    img_origin = vocdata.get_img(imgid);
    _,target = vocdata.get_annotation(imgid);
    for ob in target:
        x, y, w, h, label = ob;
        img_origin = cv2.rectangle(img_origin, (int(x - w * 0.5), int(y - h * 0.5)),
                                   (int(x + w * 0.5), int(y + h * 0.5)), (0, 0, 255), 2)
        img_origin = cv2.putText(img_origin, voc_class[label], (int(x - w * 0.5), int(y - h * 0.5) + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                 0.5, (0, 0, 0), 2)


        # xmin, ymin, xmax, ymax, label = ob;
        # img_origin = cv2.rectangle(img_origin,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),2)
        # img_origin = cv2.putText(img_origin, voc_class[label], (int(xmin), int(ymin)+ 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('original image',img_origin)
    img_trans,target = vocdata.get_item(imgid)
    img_trans *= 255
    img_trans = np.array(img_trans.clone()).astype(np.uint8);

    for ob in target:
        x, y, w, h, label = ob;
        x,y,w, h, =  x * size,y * size,w * size, h * size

        img_trans = img_trans.copy()
        img_trans = cv2.rectangle(img_trans, (np.int(x - w * 0.5), np.int(y - h * 0.5)),
                                   (np.int(x + w * 0.5), np.int(y + h * 0.5)), (0, 0, 255), 2)
        img_trans = cv2.putText(img_trans, voc_class[int(label)], (np.int(x - w * 0.5), np.int(y - h * 0.5 + 15)),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 0.5, (0, 0, 0), 2)

        # xmin, ymin, xmax, ymax, label = ob;
        # xmin, ymin, xmax, ymax = xmin * size, ymin * size, xmax * size, ymax * size
        # img_trans = img_trans.copy()
        # img_trans = cv2.rectangle(img_trans.astype(np.uint8), (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        # img_trans = cv2.putText(img_trans, voc_class[int(label)], (int(xmin), int(ymin) + 15), cv2.FONT_HERSHEY_SIMPLEX,
        #                          0.5, (0, 0, 0), 2)
    img_trans = numpy.array(img_trans).astype(np.uint8)

    cv2.imshow('transform image', img_trans)
    cv2.waitKey(0)









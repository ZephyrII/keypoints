from __future__ import annotations
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
from os.path import join as opj
from datetime import datetime
import xml.etree.ElementTree as ET

# transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([transforms.Resize([960,960]), transforms.ToTensor()])

class ChargerDataset(Dataset):
    def __init__(self, dataset_dir, num_classes, img_height, img_width, radius, transform):
        self.num_kpts = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.radius = radius         
        self.transform = transform

        self.imgs = []
        self.labels = []

        self.ann_dir = dataset_dir
        self.annotations = os.listdir(opj(dataset_dir, "annotations"))
        for a in self.annotations:
            image_file = opj(dataset_dir, 'images', a[:-4]+'.png')
            self.imgs.append(image_file)
            ann_path = opj(dataset_dir, "annotations", a)
            tree = ET.parse(ann_path)
            root = tree.getroot()
            size = root.find('size')
            # img_width = int(size.find('width').text)
            # img_height = int(size.find('height').text)
            obj = root.findall('object')[0]
            kps = obj.find('keypoints')
            img_labels = []
            for i in range(self.num_kpts):
                kp = kps.find('keypoint' + str(i))
                point_data = [
                    max(int((float(kp.find('x').text) * self.img_width)), 0),
                    max(int((float(kp.find('y').text) * self.img_height)), 0),
                    1]
                img_labels += point_data
            self.labels.append(img_labels)

        
        self.map_value = np.array([[np.linalg.norm([self.img_width - _x, self.img_height - _y]) 
                          for _x in range(img_width * 2)] for _y in range(img_height * 2)])
        
        self.offsets_x_value = np.array([[self.img_width - _x for _x in range(self.img_width * 2)] 
                                         for _y in range(self.img_height * 2)])
        self.offsets_y_value = np.array([[self.img_height - _y for _x in range(self.img_width * 2)] 
                                         for _y in range(self.img_height * 2)])
        
    def __getitem__(self, index):  
        starttime = datetime.now() 
        img = self.transform(Image.open(self.imgs[index]))
        labels = self.labels[index]

        visible = np.zeros(self.num_kpts)
        keypoints = np.zeros((self.num_kpts, 2))      
     
        maps = np.zeros((self.num_kpts, self.img_height, self.img_width), dtype='float32')
        offsets_x = np.zeros((self.num_kpts, self.img_height, self.img_width), dtype='float32')
        offsets_y = np.zeros((self.num_kpts, self.img_height, self.img_width), dtype='float32')
        
        for i in range(0, self.num_kpts * 3, 3):
            x = labels[i]
            y = labels[i + 1]
            # print(x,y)
            
            _i = i // 3

            if labels[i + 2] > 0:
                visible[_i] = 1
            else:
                visible[_i] = 0
            
            keypoints[_i][0] = x
            keypoints[_i][1] = y

            if x == 0 and y == 0:
                maps[_i] = np.zeros((self.img_height, self.img_width))
                continue
            if self.img_height - y < 0 or self.img_width - x < 0:
                continue          
            maps[_i] = self.map_value[self.img_height - y : self.img_height * 2 - y, 
                                      self.img_width  - x : self.img_width * 2  - x]       
            maps[_i][maps[_i] <= self.radius] = 1
            maps[_i][maps[_i] >  self.radius] = 0
            offsets_x[_i] = self.offsets_x_value[self.img_height - y : self.img_height * 2 - y, 
                                                 self.img_width  - x : self.img_width * 2  - x]
            offsets_y[_i] = self.offsets_y_value[self.img_height - y : self.img_height * 2 - y, 
                                                 self.img_width  - x : self.img_width * 2  - x]      
        return img, (maps, offsets_x, offsets_y), (visible, keypoints)
    
    def __len__(self):
        return len(self.labels)

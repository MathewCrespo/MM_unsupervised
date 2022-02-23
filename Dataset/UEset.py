from PIL import Image
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.cElementTree as ET
from tqdm import tqdm
import random
import pandas as pd
from random import randint,sample
import av
import numpy as np
# import cv2
# import selectivesearch

class UEset(Dataset):
    def __init__(self, root, pre_transform = None, sub_list = [0,1,2,3,4],label=True):
        self.root = root
        self.original_transform =transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        ])
        self.pre_transform = pre_transform
        self.sub_list = sub_list
        self.patient_info = []
        
        for fold in self.sub_list:
            self.scan(fold)
            

    def scan(self, fold):
        b_root = os.path.join(self.root,'{}A'.format(fold))
        m_root = os.path.join(self.root,'{}B'.format(fold))
        for img in os.listdir(m_root):
            img_path = os.path.join(m_root,img)
            label = 1.
            patient = {
                'img':img_path,
                'label':label
            }
            self.patient_info.append(patient)
        
        for img in os.listdir(b_root):
            img_path = os.path.join(b_root,img)
            label = 0.
            patient = {
                'img':img_path,
                'label':label
            }
            self.patient_info.append(patient)

    
    def __getitem__(self, idx):

        now_patient = self.patient_info[idx]
        label = now_patient['label']
        label = torch.tensor(label)  ## may be LongTensor here
        img_path = now_patient['img']
        img = Image.open(img_path)
        w, h = img.size
        img11 = img.crop((0,0,w/2,h))
        img21 = img.crop((w/2,0,w,h))
        
        img12 = self.pre_transform(img11)
        img22 = self.pre_transform(img21)

        img11 = self.original_transform(img11)
        img21 = self.original_transform(img21)

        return [img11, img21, img12, img22], label
    
    


    def __len__(self):
        return len(self.patient_info)

if __name__ == '__main__':
    pre_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    ])
    root = '/remote-home/share/MM_Ultrasound'
    data = UEset(root, pre_transform = pre_transform, sub_list = [1,2,3,4,5],label=True)
    imgs, label = data[9]
    print(imgs[1].shape)
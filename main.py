#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from utils.logger import Logger
from torch.optim import Adam
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,4,5'
from collections import OrderedDict
from Dataset.UEset import UEset
from models.encoder import Res_Encoder, Vgg_Encoder
from trainers.BaseTrainer import BaseTrainer
from trainers.MMTrainer import MMTrainer



class Base_config(object):
    ## for baseline model
    def __init__(self, log_root, args):
        #self.net = getattr(import_module('models.graph_attention'),args.net)(t=args.t, task=args.task)
        #print(self.net)
        #device_ids = range(torch.cuda.device_count())
        self.net = Vgg_Encoder()
        self.net = self.net.cuda()
        self.net = nn.DataParallel(self.net,device_ids=range(torch.cuda.device_count()))
        self.train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)
        self.trainbag = UEset(args.data_root, pre_transform = self.train_transform, sub_list=[x for x in [1,2,3,4,5] if x!=args.test_fold])
        self.train_loader = DataLoader(self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
        self.trainer = BaseTrainer(self.net, self.optimizer, self.lrsch, self.train_loader, self.logger, 10)
        self.save_config(args)

    def save_config(self,args):
        config_file = './saved_configs/'+args.log_root+'.txt'
        f = open(config_file,'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])

class MM_config(object):
    ## for baseline model
    def __init__(self, log_root, args):
        #self.net = getattr(import_module('models.graph_attention'),args.net)(t=args.t, task=args.task)
        #print(self.net)
        #device_ids = range(torch.cuda.device_count())
        self.net = Vgg_Encoder()
        self.net = self.net.cuda()
        self.net = nn.DataParallel(self.net,device_ids=range(torch.cuda.device_count()))
        self.train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)
        self.trainbag = UEset(args.data_root, pre_transform = self.train_transform, sub_list=[x for x in [1,2,3,4,5] if x!=args.test_fold])
        self.train_loader = DataLoader(self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
        self.trainer = MMTrainer(self.net, self.optimizer, self.lrsch, self.train_loader, self.logger, 10)
        self.save_config(args)

    def save_config(self,args):
        config_file = './saved_configs/'+args.log_root+'.txt'
        f = open(config_file,'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])




if __name__=='__main__':
    #configs = getattr(import_module('configs.'+args.config),'Config')()
    #configs = configs.__dict__
    parser = argparse.ArgumentParser(description='Multi-modal Unsupervised Framework')
    parser.add_argument('--data_root',type=str,default='/remote-home/share/MM_Ultrasound')
    parser.add_argument('--gpu',type=str)
    parser.add_argument('--log_root',type=str)
    parser.add_argument('--test_fold',type=int,default=0, help='which fold of data is used for test')
    parser.add_argument('--lr',type=float,default=0.03)
    parser.add_argument('--epoch',type=int,default=50)
    parser.add_argument('--resume',type=int,default=-1)
    parser.add_argument('--batchsize',type=int,default=256)
    parser.add_argument('--net',type=str,default='H_Attention_Graph')
    parser.add_argument('--base', type = str, default='base')
    # parse parameters
    args = parser.parse_args()
    
    log_root = os.path.join('/remote-home/huhongyu/experiments/MMU/',args.log_root)
    if not os.path.exists(log_root):
        os.mkdir(log_root)

    # notice here
    if args.base == 'base':
        config_object = Base_config(log_root,args)
    else:
        config_object = MM_config(log_root,args)
    

    # train and eval
    for epoch in range(config_object.logger.global_step, args.epoch):
        print('Now epoch {}'.format(epoch+1))
        config_object.trainer.train(epoch=epoch+1)
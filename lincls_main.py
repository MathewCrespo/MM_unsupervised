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
from collections import OrderedDict
from Dataset.UEset import UEset
from models.encoder import Res_Encoder, classifer, Vgg_Encoder
from trainers.BaseTrainer import BaseTrainer
from trainers.LinclsTrainer import LinclsTrainer




class lincls_config(object):
    ## for baseline model
    def __init__(self, log_root, model_path, args):
        #self.net = getattr(import_module('models.graph_attention'),args.net)(t=args.t, task=args.task)
        #print(self.net)

        # load pretrain model and freeze 
        self.encoder = Vgg_Encoder()
        pretrain_model = torch.load(model_path)['net']
        new_dict = OrderedDict()
        for k,v in pretrain_model.items():
            new_k = k[7:]
            new_dict[new_k] = v
        self.encoder.load_state_dict(new_dict)
        for name, v in self.encoder.named_parameters():
            v.requires_grad = False
        self.encoder = self.encoder.cuda()
        self.encoder.eval()

        # add cls layer
        self.cls = classifer()
        self.cls.fc.weight.data.normal_(mean=0.0,std=0.01)
        self.cls.fc.bias.data.zero_()
        self.cls = self.cls.cuda()

        self.train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])

        self.test_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
        ])
        '''
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        '''
        self.optimizer = torch.optim.SGD(self.cls.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)
        self.trainbag = UEset(args.data_root, pre_transform = self.train_transform, sub_list=[x for x in [1,2,3,4,5] if x!=args.test_fold])
        self.testbag = UEset(args.data_root, pre_transform = self.test_transform, sub_list=[args.test_fold])
        self.train_loader = DataLoader(self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
        self.test_loader = DataLoader(self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
        # load trainer changes
        self.trainer = LinclsTrainer(self.encoder, self.cls, self.optimizer, self.lrsch, self.train_loader, self.test_loader, self.logger, 10)
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
    parser = argparse.ArgumentParser(description='Linear Classification for Multi-modal Unsupervised')
    parser.add_argument('--data_root',type=str,default='/remote-home/share/MM_Ultrasound')
    parser.add_argument('--pretrained',type=str,default='/remote-home/huhongyu/experiments/MMU/dense1_fm4_44/ckp/net.ckpt200.pth')
    parser.add_argument('--log_root',type=str)
    parser.add_argument('--test_fold',type=int,default=0, help='which fold of data is used for test')
    parser.add_argument('--lr',type=float,default=0.03)
    parser.add_argument('--epoch',type=int,default=50)
    parser.add_argument('--resume',type=int,default=-1)
    parser.add_argument('--batchsize',type=int,default=256)
    parser.add_argument('--net',type=str,default='H_Attention_Graph')

    # parse parameters
    args = parser.parse_args()
    log_root = os.path.join('/remote-home/huhongyu/experiments/MMU_lincls/',args.log_root)
    if not os.path.exists(log_root):
        os.mkdir(log_root)

    model_path = '/remote-home/huhongyu/experiments/MMU/{}/ckp/net.ckpt200.pth'.format(args.pretrained)

    
    config_object = lincls_config(log_root, model_path, args)
    

    # train and eval
    for epoch in range(config_object.logger.global_step, args.epoch):
        print('Now epoch {}'.format(epoch+1))
        config_object.trainer.train(epoch=epoch+1)
        config_object.trainer.test(epoch=epoch+1)
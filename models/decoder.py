# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:11:41 2020
@author: Administrator
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def l2_normal(x,dim=1):
    norm = x.pow(2).sum(dim, keepdim=True).pow(1./2)+1e-10
    out = x.div(norm)
    return out
            
def l1_normal(x,dim=1):
    x=x+10e-10
    norm = x.sum(dim, keepdim=True)
    out = x.div(norm)
    return out  

class decoder(nn.Module):
    def __init__(self,channel_list=[128,128,64,32,16,32,8]):
        super(decoder, self).__init__()
        self.upsample=nn.Upsample(scale_factor=(2,2))
        
        
        self.conv_layer8x=nn.Sequential(
                                              nn.Conv2d(in_channels=channel_list[0]+channel_list[1],out_channels=channel_list[1],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[1]),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=channel_list[1],out_channels=channel_list[2],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[2]),
                                              nn.ReLU(True))

        self.conv_layer4x=nn.Sequential(
                                              nn.Conv2d(in_channels=channel_list[2]*2,out_channels=channel_list[2],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[2]),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=channel_list[2],out_channels=channel_list[3],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[3]),
                                              nn.ReLU(True)
                                                )
        self.conv_layer2x=nn.Sequential(
                                              nn.Conv2d(in_channels=channel_list[3]*2,out_channels=channel_list[3],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[3]),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=channel_list[3],out_channels=channel_list[4],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[4]),
                                              nn.ReLU(True)
                                                )
        self.embedding_branch=nn.Sequential(
                                              nn.Conv2d(in_channels=channel_list[4]*2,out_channels=channel_list[5],kernel_size=(3,3),padding=(1,1)),
                                              nn.BatchNorm2d(channel_list[5]),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=channel_list[5],out_channels=channel_list[5],kernel_size=(3,3),padding=(1,1))
                                                )
        if len(channel_list)>6:
            self.clustering_branch=nn.Sequential(
                                                  nn.Conv2d(in_channels=channel_list[4]*2,out_channels=channel_list[6],kernel_size=(3,3),padding=(1,1)),
                                                  nn.BatchNorm2d(channel_list[6]),
                                                  nn.ReLU(True),
                                                  nn.Conv2d(in_channels=channel_list[6],out_channels=channel_list[6],kernel_size=(3,3),padding=(1,1)),
                                                  nn.Sigmoid()
                                                )
    def forward(self, x,region_output_size=(4,4),if_clustering=False,if_test=False):
        features=self.upsample(x[-1])
        features=torch.cat([features,x[-2]],dim=1)
        features=self.conv_layer8x(features)
        features=self.upsample(features)
        features=torch.cat([features,x[-3]],dim=1)
        features=self.conv_layer4x(features)
        features=self.upsample(features)
        features=torch.cat([features,x[-4]],dim=1)
        features=self.conv_layer2x(features)
        features=self.upsample(features)
        features=torch.cat([features,x[-5]],dim=1)
        del x
        features=self.embedding_branch(features)
        
        features=l2_normal(features)
        patch_features=nn.functional.adaptive_avg_pool2d(features,output_size=region_output_size)

        return l2_normal(patch_features)
    

# def loss_entropy(attention_map):
#     log_attention_map=torch.log2(attention_map+ 1e-30)
#     entropy=(-attention_map*log_attention_map).mean()
#     return entropy

def loss_entropy(attention_map):
    entropy=(-attention_map*torch.log2(attention_map+ 1e-30)-(1-attention_map)*torch.log2(1-attention_map+ 1e-30)).mean()
    return entropy
    
def loss_distance(feature_map1,feanture_map2):
    return (feature_map1*feanture_map2).sum(1).mean()
    
def loss_area(attention_map,limit_rate=1/16,if_l2=True):
    area_list=torch.mean(attention_map,(2,3))-limit_rate
    if if_l2:
        loss=((-area_list)**2).to(torch.float32).mean()
    else:
        loss=torch.abs(area_list).to(torch.float32).mean()
    return loss
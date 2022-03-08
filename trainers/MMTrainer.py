#!/usr/bin/env
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
# for debug
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from models.loss import Global_Loss, Dense_Loss
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class MMTrainer(object):
    def __init__(self, net, optimizer, lrsch, train_loader, logger,
                 save_interval=1):
        self.net = net
        self.optimizer = optimizer
        self.lrsch = lrsch
        self.train_loader = train_loader
        self.logger = logger
        self.logger.global_step = 0
        self.save_interval = save_interval
        self.loss1 = Global_Loss(aug=True,domain=True) # for img-level global loss
        self.loss2 = Dense_Loss(patch_size=(4,4), h=[4])# for fine-grained local structure loss


            
    def train(self,epoch):
        epoch = epoch
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        self.net.train()
        end = time.time()
        self.logger.update_step()
        p_all = 0.
        d_all = 0.
        for img, _ in (tqdm(self.train_loader, ascii=True, ncols=60)): # we do not use img label in unsupervised pretrain
            # reset gradients
            data_time.update(time.time()-end) 
            self.optimizer.zero_grad()

            img = img.cuda()
            _, f_list, f = self.net(img)  # output from the encoder
            
            patient_loss, target, f11, f21, logits = self.loss1(f) # img-level global loss
            dense_loss = self.loss2(f_list)
            patient_f = torch.cat([f11,f21], dim=1)
            loss = patient_loss + dense_loss
            acc1, acc5 = self.accuracy(logits, target, topk=(1, 5)) # notice here  
            # loss sum
            p_all += patient_loss.item()
            d_all += dense_loss.item()

            del target
            del logits
            

            losses.update(loss.item(),img.size(0))
            top1.update(acc1[0], img.size(0))
            top5.update(acc5[0], img.size(0))
        
            # backward pass
            del f_list
            del f11
            del f21
            del img
            loss.backward()
            # step
            self.optimizer.step()
            self.lrsch.step()
            
            batch_time.update(time.time()-end)
            end = time.time()

            progress.display(1)
        #self.log_metric("Train", target, prob, pred)

        self.log_loss("Train", p_all/len(self.train_loader), d_all/len(self.train_loader))

        if not (self.logger.global_step % self.save_interval):
            self.logger.save(self.net, self.optimizer, self.lrsch, self.loss1)

        #print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(self.epoch, train_loss.cpu().numpy()[0], train_error))
    
    def log_loss(self, prefix, patient_loss_all, dense_loss_all):

        self.logger.log_scalar(prefix+'/'+'Patient', patient_loss_all, print=True)
        self.logger.log_scalar(prefix+'/'+'Dense', dense_loss_all, print= True)

    def accuracy(self, output, target, topk=(1,)):
    #Computes the accuracy over the k top predictions for the specified values of k
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

        

        
    def log_metric(self, prefix, target, prob, pred):
        pred_list = np.concatenate(pred)
        prob_list = np.concatenate(prob)
        target_list = np.concatenate(target)
        cls_report = classification_report(target_list, pred_list, output_dict=True, zero_division=0)
        acc = accuracy_score(target_list, pred_list)
        #print ('acc is {}'.format(acc))
        auc_score = roc_auc_score(target_list, prob_list)
        print('auc is {}'.format(auc_score))
        #print(cls_report)

        self.logger.log_scalar(prefix+'/'+'AUC', auc_score, print=True)
        self.logger.log_scalar(prefix+'/'+'Acc', acc, print= True)
        self.logger.log_scalar(prefix+'/'+'Malignant_precision', cls_report['1']['precision'], print= True)
        self.logger.log_scalar(prefix+'/'+'Benign_precision', cls_report['0']['precision'], print= True)
        self.logger.log_scalar(prefix+'/'+'Malignant_recall', cls_report['1']['recall'], print= True)
        self.logger.log_scalar(prefix+'/'+'Benign_recall', cls_report['0']['recall'], print= True)
        self.logger.log_scalar(prefix+'/'+'Malignant_F1', cls_report['1']['f1-score'], print= True)
        

if __name__ == '__main__':
    # for debugging training function

    parser = argparse.ArgumentParser(description='Ultrasound CV Framework')
    parser.add_argument('--config',type=str,default='grey_SWE')

    args = parser.parse_args()
    configs = getattr(import_module('configs.'+args.config),'Config')()
    configs = configs.__dict__
    logger = configs['logger']
    logger.auto_backup('./')
    logger.backup_files([os.path.join('./configs',args.config+'.py')])
    trainer = configs['trainer']
    for epoch in range(logger.global_step, configs['epoch']):
        trainer.train()
        trainer.test()




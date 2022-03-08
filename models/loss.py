import torch
import torch.nn as nn
import torch.nn.functional as F

class Global_Loss(nn.Module):
    def __init__(self, t=0.07,aug=True,domain=True):
        super(Global_Loss, self).__init__()
        self.t = t
        self.aug = aug
        self.domain = domain
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, f):
        f = F.normalize(f,dim=2)
        f11 = f[:,0,:]
        f21 = f[:,1,:]
        f12 = f[:,2,:]
        f22 = f[:,3,:]
        l = torch.tensor(0.).cuda()
        logits = []
        if self.aug:
            l_pos_aug1 = torch.einsum('nc,nc->n',[f11,f12]).unsqueeze(-1)
            l_pos_aug2 = torch.einsum('nc,nc->n',[f21,f22]).unsqueeze(-1)
            l_neg_aug1 = torch.einsum('nc,ck->nk',[f11,f12.T])
            l_neg_aug2 = torch.einsum('nc,ck->nk',[f21,f22.T])
            logits_aug1 = torch.cat([l_pos_aug1,l_neg_aug1],dim=1)
            logits_aug1 /= self.t
            logits_aug2 = torch.cat([l_pos_aug2,l_neg_aug2],dim=1)
            logits_aug2 /= self.t
            labels = torch.zeros(logits_aug1.shape[0],dtype=torch.long).cuda()
            aug_loss = self.criterion(logits_aug1,labels) + self.criterion(logits_aug2,labels)
            l += aug_loss
            logits.append(logits_aug1)
            logits.append(logits_aug2)
        
        if self.domain:
            l_pos_d1 = torch.einsum('nc,nc->n',[f11,f21]).unsqueeze(-1)
            l_pos_d2 = torch.einsum('nc,nc->n',[f12,f22]).unsqueeze(-1)
            l_neg_d1 = torch.einsum('nc,ck->nk',[f11,f21.T])
            l_neg_d2 = torch.einsum('nc,ck->nk',[f12,f22.T])
            logits_d1 = torch.cat([l_pos_d1,l_neg_d1],dim=1)  # n*(k+1)
            logits_d1 /= self.t
            logits_d2 = torch.cat([l_pos_d2,l_neg_d2],dim=1)
            logits_d2 /= self.t
            labels = torch.zeros(logits_d1.shape[0],dtype=torch.long).cuda()
            d_loss = self.criterion(logits_d1,labels) + self.criterion(logits_d2,labels)
            l += d_loss
            logits.append(logits_d1)
            logits.append(logits_d2)
        stack_logits = torch.stack([logit for logit in logits],dim=0)
        mean_logits = torch.mean(stack_logits,dim=0)
        #print(mean_logits.shape)
        #return l, aug_loss, d_loss, labels, f11, f21, mean_logits
        return l, labels, f11, f21, mean_logits

class Dense_Loss(nn.Module):
    def __init__(self,patch_size,h=[0,1,2,3,4]):
        super(Dense_Loss, self).__init__()
        self.h = h
        self.patch_size = patch_size

    def forward(self, f):
        all_loss = torch.tensor(0.).cuda()
        # choose which hierachies of feature map are used. 
        for i in self.h:
            feature_map = f[i]  # feature map b' x c x h x w
            feature_map = F.normalize(feature_map,dim=1) ## l2 norm first
            b,c,w,h = feature_map.shape
            feature_map = feature_map.view(-1,4,c,w,h) # reshape the feature map
            fm11 = feature_map[:,0,:,:]
            fm21 = feature_map[:,1,:,:]
            patch11 = F.adaptive_avg_pool2d(fm11,output_size=self.patch_size) # b x c x n x n
            patch21 = F.adaptive_avg_pool2d(fm21,output_size=self.patch_size)
            #print(patch11.shape)
            # now the feature map in each modality is b x c x n x n (n denotes patch size)
            patch_flatten11 = patch11.view(int(b/4),c,-1)  # b x c x n^2
            patch_flatten21 = patch21.view(int(b/4),c,-1)
            # b x c x n^2
            sim_matrix11 = torch.einsum('abc,abd->acd',[patch_flatten11,patch_flatten11]) # b x n^2 x n^2
            sim_matrix21 = torch.einsum('abc,abd->acd',[patch_flatten21,patch_flatten21])
            sim_flatten11 = sim_matrix11.view(sim_matrix11.shape[0],-1)
            sim_flatten21 = sim_matrix21.view(sim_matrix21.shape[0],-1)  # b x n^4
            patch_loss = F.pairwise_distance(sim_flatten11,sim_flatten21,p=2)
            h_loss = torch.mean(patch_loss)
            all_loss += h_loss
            del feature_map
            del fm11
            del fm21
            del patch11
            del patch21
            del sim_matrix11
            del sim_matrix21
            del sim_flatten11
            del sim_flatten21
        return all_loss

class Img_Loss(nn.Module):
    def __init__(self, t=0.01):
        super(Img_Loss, self).__init__()
        self.t = t
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, f):
        f = F.normalize(f,dim=2)
        f11 = f[:,0,:]
        f21 = f[:,1,:]
        f12 = f[:,2,:]
        f22 = f[:,3,:]
        l = torch.tensor(0.).cuda()
        logits = []
        if self.aug:
            l_pos_aug1 = torch.einsum('nc,nc->n',[f11,f12]).unsqueeze(-1)
            l_pos_aug2 = torch.einsum('nc,nc->n',[f21,f22]).unsqueeze(-1)
            l_neg_aug1 = torch.einsum('nc,ck->nk',[f11,f12.T])
            l_neg_aug2 = torch.einsum('nc,ck->nk',[f21,f22.T])
            logits_aug1 = torch.cat([l_pos_aug1,l_neg_aug1],dim=1)
            logits_aug1 /= self.t
            logits_aug2 = torch.cat([l_pos_aug2,l_neg_aug2],dim=1)
            logits_aug2 /= self.t
            labels = torch.zeros(logits_aug1.shape[0],dtype=torch.long).cuda()
            aug_loss = self.criterion(logits_aug1,labels) + self.criterion(logits_aug2,labels)
            l += aug_loss
            logits.append(logits_aug1)
            logits.append(logits_aug2)
        
        if self.domain:
            l_pos_d1 = torch.einsum('nc,nc->n',[f11,f21]).unsqueeze(-1)
            l_pos_d2 = torch.einsum('nc,nc->n',[f12,f22]).unsqueeze(-1)
            l_neg_d1 = torch.einsum('nc,ck->nk',[f11,f21.T])
            l_neg_d2 = torch.einsum('nc,ck->nk',[f12,f22.T])
            logits_d1 = torch.cat([l_pos_d1,l_neg_d1],dim=1)  # n*(k+1)
            logits_d1 /= self.t
            logits_d2 = torch.cat([l_pos_d2,l_neg_d2],dim=1)
            logits_d2 /= self.t
            labels = torch.zeros(logits_d1.shape[0],dtype=torch.long).cuda()
            d_loss = self.criterion(logits_d1,labels) + self.criterion(logits_d2,labels)
            l += d_loss
            logits.append(logits_d1)
            logits.append(logits_d2)
        stack_logits = torch.stack([logit for logit in logits],dim=0)
        mean_logits = torch.mean(stack_logits,dim=0)
        #print(mean_logits.shape)
        return l, labels, f11, f21, mean_logits


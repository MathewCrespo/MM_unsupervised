import torch
import torch.nn as nn
import torch.nn.functional as F

class Global_Loss(nn.Module):
    def __init__(self, t=0.01,aug=True,domain=False):
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
        return l, labels, f11, f21, mean_logits

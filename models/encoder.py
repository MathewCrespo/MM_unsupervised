from .resnet import ResNet10
from .vgg import vgg16_bn_tiny
import torch
import torch.nn as nn
import torch.nn.functional as F

class Res_Encoder(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(Res_Encoder, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor_part1 = ResNet10()

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.L),
            nn.ReLU(),
        )

    def forward(self, x, debug=False):

        batch = x.shape[0]
        x = x.view(-1,3,224,224)
        #print(x.shape)
        f = self.feature_extractor_part1(x)
        f = f.view(-1,512*7*7)
        #print(f.shape)
        
        f = self.feature_extractor_part2(f)
        f = f.view(batch,4,-1)

        return f

class Vgg_Encoder(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(Vgg_Encoder, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor_part1 = vgg16_bn_tiny()

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(128 * 14 * 14, self.L),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, debug=False):

        batch = x.shape[0]
        x = x.view(-1,3,224,224)
        #print(x.shape)
        f = self.feature_extractor_part1(x)  # feature maps at 5 scales
        final_f = f[-1].view(-1,128*14*14)
        final_f = self.feature_extractor_part2(final_f)
        final_f = final_f.view(batch,4,-1)

        return torch.tensor(batch,device=x.device), f, final_f

class classifer(nn.Module):
    def __init__(self):
        super(classifer,self).__init__()
        '''
        self.cls_layer = nn.Sequential(
            nn.Linear(1000,128),
            nn.ReLU(),
            nn.Linear(128,2),
            #nn.Sigmoid()
        )
        '''
        self.fc = nn.Linear(1000,2)
    def forward(self,f):
        #probs = self.cls_layer(f)
        probs = self.fc(f)
        return probs

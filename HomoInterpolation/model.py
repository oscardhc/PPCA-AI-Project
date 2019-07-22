import torch
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, feature_num):
        super(Discriminator, self).__init__()
        self.feat_n = feature_num
        self.dis = nn.Sequential(
            nn.Conv2d(512, 256, 1), 
            nn.InstanceNorm2d(256, affine=True), 
            nn.LeakyReLU(), 
            nn.Conv2d(256, 256, 4, 2, 1), 
            nn.InstanceNorm2d(256, affine=True), 
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1), 
            nn.InstanceNorm2d(256, affine=True), 
            nn.LeakyReLU(), 
            nn.Conv2d(256, 256, 2, 1)
        )
        self.critic = nn.Conv2d(256, 256, 1, 1)
        self.homo = nn.Conv2d(256, self.feat_n, 1, 1)

    def forward(self, feat):
        tmp = self.dis(feat)
        crit = self.critic(tmp)
        hom = self.homo(tmp)
        return crit, hom


class Encoder(nn.Module):
    def __init__(self, useVGG=True):
        super(Encoder, self).__init__()

        if useVGG:
            self.model = VGG()
        else:
            model = []
            size = 128
            ch = [3, 64, 128, 256, 512]
            dep = [2, 2, 4, 4]
            for i in range(4):
                model = model + [
                    nn.Conv2d(ch[i], ch[i + 1], 3, 1, 1),
                    nn.ReLU()
                ]
                for _ in range(dep[i] - 1):
                    model = model + [
                        nn.Conv2d(ch[i + 1], ch[i + 1], 3, 1, 1),
                        nn.ReLU()
                    ]
                model = model + [nn.MaxPool2d((2, 2))]
            model = model + [nn.Conv2d(512, 512, 3, 1, 1)]
            self.model = nn.Sequential(*model)
    
    def forward(self, image):
        x = self.model(image)
        return x


class Interp(nn.Module):
    def __init__(self, feature_num):
        super(Interp, self).__init__()
        self.feat_n = feature_num
        self.interp_set = []
        for i in range(feature_num):
            model = []
            for _ in range(2):
                model = model + [
                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
            model = model + [nn.Conv2d(512, 512, 3, 1, 1)]
            self.interp_set = self.interp_set + [nn.sequential(*model)]
        
    def forward(self, fA, fB, str):
        x = fA
        for i in range(self.feat_n):
            x = x + self.interp_set[i](fB - fA)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        model = []
        ch = [512, 512, 256, 128, 64]
        dep = [1, 3, 3, 1]
        for i in range(4):
            model = model + [
                nn.Conv2d(ch[i], ch[i + 1], 3, 1, 1),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU()
            ]
            for _ in range(dep[i] - 1):
                model = model + [
                    nn.Conv2d(ch[i + 1], ch[i + 1], 3, 1, 1),
                    nn.BatchNorm2d(ch[i + 1]),
                    nn.ReLU()
                ]
            model = model + [nn.Upsample(scale_factor=2),
                nn.Conv2d(ch[i + 1], ch[i + 1], 3, 1, 1),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU()
            ]

        model = model + [nn.Conv2d(64, 3, 3, 1, 1)]

        self.dec = nn.Sequential(*model)

    def forward(self, f):
        x = self.dec(f)
        return x


class KG(nn.Module):
    def __init__(self):
        super(KG, self).__init__()
        self.model = nn.Conv2d(512, 512, 1, 1)
    
    def forward(self, feat):
        x = self.model(feat)
        return x


class VGG(nn.Module):

    def __init__(self, path = 'checkpoints/vgg.model'):
        self.model = torch.load(path)

    def forward(self, x):
        return self.model(x)


import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
from collections import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, attr):
        super(Discriminator, self).__init__()
        self.attr_n = attr_n
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
        self.homo = []
        for att in attr:
            self.homo += [nn.Conv2d(256, att, 1, 1)]

    def forward(self, feat):
        tmp = self.dis(feat)
        crit = self.critic(tmp)
        hom = []
        for hom_net in self.homo:
            temp = hom_net(tmp).squeeze()
            hom += [temp]
        return crit, hom


class Encoder(nn.Module):

    def __init__(self, path):
        super(Encoder, self).__init__()
        self.model = VGG(path)
        # print(self.state_dict())

    def forward(self, image):
        x = self.model(image)
        return x


class Interp(nn.Module):
    def __init__(self, attr_n):
        super(Interp, self).__init__()
        self.attr_n = attr_n
        self.interp_set = []
        for i in range(attr_n):
            model = []
            for _ in range(2):
                model = model + [
                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
            model = model + [nn.Conv2d(512, 512, 3, 1, 1)]
            self.interp_set = self.interp_set + [nn.Sequential(*model)]

        self.interp_set = nn.ModuleList(self.interp_set)
        
    def forward(self, fA, fB, strenth):
        x = fA
        strenth = strenth.unsqueeze(2).unsqueeze(3).expand((-1, -1, fA.size(2), fA.size(3)))
        for i in range(self.attr_n):
            x = x + self.interp_set[i](fB - fA) * strenth[:, i:i + 1, :, :]
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
        self.model = nn.Conv2d(512, 512, 1)
    
    def forward(self, feat):
        x = self.model(feat)
        return x


class VGG(nn.Module):

    def __init__(self, path):
        super(VGG, self).__init__()
        model = OrderedDict()
        size = 128
        ch = [3, 64, 128, 256, 512]
        dep = [2, 2, 4, 4]
        for i in range(4):
            model.update(OrderedDict([
                ('conv%d_%d' % (i+1, 1), nn.Conv2d(ch[i], ch[i + 1], 3, 1, 1)),
                ('relu%d_%d' % (i+1, 1), nn.ReLU(inplace=True))
            ]))
            for j in range(dep[i] - 1):
                model.update(OrderedDict([
                    ('conv%d_%d' % (i+1, j+2), nn.Conv2d(ch[i + 1], ch[i + 1], 3, 1, 1)),
                    ('relu%d_%d' % (i+1, j+2), nn.ReLU(inplace=True))
                ]))
            model.update(OrderedDict([('pool%d' % (i+1), nn.MaxPool2d((2, 2)))]))
        model.update(OrderedDict([
            ('conv5_1', nn.Conv2d(512, 512, 3, 1, 1)),
            ('relu5_1', nn.ReLU(inplace=True))
        ]))
        self.model = nn.Sequential(model)

        sdict = torch.load(path)
        mdict = self.state_dict()

        def contains(x):
            x = x[6:]
            for i in sdict.keys():
                if i.find(x) != -1:
                    return sdict[i]

        sdict = {key: contains(key) for key, value in mdict.items()}
        self.load_state_dict(sdict)

    def forward(self, x):
        return self.model(x)


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):

    def __init__(self, input_size, input_channel, condition_num):

        super(Discriminator, self).__init__()
        model = []
        depth = np.log2(input_size)
        cur_channel = 64
        for i in range(depth):
            '''normalization?'''
            model += [
                nn.Conv2d(input_channel if i == 0 else cur_channel // 2,
                          cur_channel, 4, 2, 1),
                nn.LeakyReLU(inplace=True)
            ]
        self.model = nn.Sequential(*model)
        self.last = nn.Conv2d(cur_channel // 2, 1, 1)
        self.cond = nn.Conv2d(cur_channel // 2, condition_num, 1)

    def forward(self, x):
        y = self.model(x)
        return self.last(y), self.cond(y)


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c)
        )

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):

    def __init__(self, ic, cond_num):
        super(Generator, self).__init__()
        ch = 16
        tm = 2

        model = [
            nn.Conv2d(ic + cond_num, ch, ic * 2 + 1, padding=ic)
        ]
        for _ in range(tm):
            model = model + [
                nn.Conv2d(ch, ch * 2, 4, 2, 1),
                nn.BatchNorm2d(ch * 2),
                nn.ReLU(inplace=True)
            ]
            ch = ch * 2

        for _ in range(5):
            model = model + [
                ResBlock(ch)
            ]

        for _ in range(tm):
            model = model + [
                nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True)
            ]
            ch = ch // 2

        self.model = nn.Sequential(*model)
        self.out_image = nn.Sequential(*[
            nn.Conv2d(ch, ic, ic * 2 + 1, padding=ic),
            nn.Tanh()
        ])
        self.out_attention = nn.Sequential(*[
            nn.Conv2d(ch, 1, ic * 2 + 1, padding=ic),
            nn.Sigmoid()
        ])


    def forward(self, x, cond):
        torch.reshape(cond, (cond.size(0), cond.size(1), 1, 1))
        cond = cond.expand(cond.size(0), cond.size(1), x.size(2), x.size(3))
        input = torch.cat((x, cond), dim=1)
        y = self.model(input)
        return self.out_image(y), self.out_attention(y)
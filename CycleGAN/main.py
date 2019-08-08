import glob
import random
import os

import torch
from PIL import Image
import numpy as np
import csv
import torch.utils.data

def getTestImages(path, size=128):
    files = sorted(glob.glob((path + '/*.*')))
    print(path, files)
    ret = []
    for file in files:
        img = Image.open(file)
        W, H = img.size
        bnd = (H - W) // 2
        img = img.crop((0, bnd, W, W))
        img = img.resize((size, size), Image.ANTIALIAS)
        ret += [(np.array(img) / 127.5 - 1.0).transpose(2, 0, 1)]
    return ret

class pixDataset(torch.utils.data.Dataset):
    
    def __init__(self, num, path, attr):
        super(pixDataset, self).__init__()
        self.path = path
        self.num = num
        self.picSize = 128
        self.fileA = []
        self.oritA = []
        self.fileB = []
        self.oritB = []
        
        with open(path + '/celeba-with-orientation.csv') as f:
            info = csv.DictReader(f)
            for row in info:
                if row[attr] == '-1':
                    self.fileA.append(row['name'])
                    self.oritA.append(row['orientation'])
                else:
                    self.fileB.append(row['name'])
                    self.oritB.append(row['orientation'])
                if len(self.fileA) + len(self.fileB) == self.num:
                    break
                    
        print(len(self.fileA), len(self.fileB))
        
    def __len__(self):
        return max(len(self.fileA), len(self.fileB)) // 256 * 256

    def getImage(self, index, flag):
        if flag:
            img = Image.open(self.path + '/' + self.fileA[index])
        else:
            img = Image.open(self.path + '/' + self.fileB[index])
        
        W, H = img.size
        bnd = (H - W) // 2
        img = img.crop((0, bnd, W, W))

        if flag and self.oritA[index] == 'left':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if not flag and self.oritB[index] == 'left':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize((self.picSize, self.picSize), Image.ANTIALIAS)
        return (np.array(img) / 127.5 - 1.0).transpose(2, 0, 1)

    def __getitem__(self, item):
        return (self.getImage(item % len(self.fileA), True), self.getImage(item % len(self.fileB), False))
    
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

class ResBlock(nn.Module):
    
    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3),
            nn.BatchNorm2d(c)
        )
        
    def forward(self, x):
        return x + self.model(x)

class Generator(nn.Module):
    
    def __init__(self, sz, ic):
        super(Generator, self).__init__()
        ch = 16
        tm = 3
        
        model = [
            nn.ReflectionPad2d(ic),
            nn.Conv2d(ic, ch, ic * 2 + 1)
        ]
        for _ in range(tm):
            model = model + [
                nn.Conv2d(ch, ch * 2, 4, 2, 1),
                nn.BatchNorm2d(ch * 2),
                nn.ReLU(inplace=True)
            ]
            ch = ch * 2
        
        for _ in range(9):
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
        model = model + [
            nn.ReflectionPad2d(ic),
            nn.Conv2d(ch, ic, ic * 2 + 1),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    
    def __init__(self, sz, ic):
        super(Discriminator, self).__init__()
        ch = 16
        
        model = [
            nn.Conv2d(ic, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        sz = sz // 2
        
        while sz > 8:
            model = model + [
                nn.Conv2d(ch, ch * 2, 4, 2, 1),
                nn.BatchNorm2d(ch * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            ch = ch * 2
            sz = sz // 2
        
        model = model + [
            nn.Conv2d(ch, 1, 4, 2, 1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
        
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    

def main():
    
    dv = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    bs = 128
    print(dv)
    
    GAB = Generator(128, 3).to(dv)
    GBA = Generator(128, 3).to(dv)
    DA = Discriminator(128, 3).to(dv)
    DB = Discriminator(128, 3).to(dv)
#     GAB.apply(weights_init)
#     GBA.apply(weights_init)
#     DA.apply(weights_init)
#     DB.apply(weights_init)

    path = '../../young'

    GAB.load_state_dict(torch.load('%s/gab.pth' % path))
    GBA.load_state_dict(torch.load('%s/gba.pth' % path))
    DA.load_state_dict(torch.load('%s/da.pth' % path))
    DB.load_state_dict(torch.load('%s/db.pth' % path))
    with torch.no_grad():
        ts = torch.tensor(getTestImages('../../img1')).float().to(dv)
        re = ((GAB(ts).cpu().numpy() + 1) * 127.5).astype('uint8')
        for i, img in enumerate(re):
            im = Image.fromarray(img.transpose(1, 2, 0))
            im.save('../../img1/out%d.jpg' % i)
    return
#     wr = SummaryWriter('./run', flush_secs=3)

    
    exit()
    
    ds = pixDataset(192000, '../../align-celeba/dat/celeba-dataset', 'Male')
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
    
    MS = nn.MSELoss()
    L1 = nn.L1Loss()
    
    oG = optim.Adam(itertools.chain(GAB.parameters(), GBA.parameters()), lr=2e-4, betas=(0.5, 0.999))
    oDA = optim.Adam(DA.parameters(), lr=2e-4, betas=(0.5, 0.999))
    oDB = optim.Adam(DB.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    lr = torch.Tensor(np.zeros((bs, 1, 4, 4))).to(dv).fill_(1)
    lf = torch.Tensor(np.zeros((bs, 1, 4, 4))).to(dv).fill_(0)
    
    for ep in range(9):
        print(len(dl))
        for it, dt in enumerate(dl):
            
            ra = dt[0].float().to(dv)
            rb = dt[1].float().to(dv)

            GAB.train()
            GBA.train()
            oG.zero_grad()

            fb = GAB(ra)
            fa = GBA(rb)

            lsG1 = L1(fa, ra) + L1(fb, rb)
            lsG2 = MS(DA(fa), lr) + MS(DB(fb), lr)
            lsG3 = L1(GAB(fa), rb) + L1(GBA(fb), ra)

            lsG =  lsG1 * 0.5 + lsG2 * 2.0 + lsG3 * 5.0
            lsG.backward()
            oG.step()
            
            oDA.zero_grad()
            rA = DA(ra)
            fA = DA(fa.detach())
            lsDA = MS(rA, lr)
            lsDA = lsDA + MS(fA, lf)
            lsDA.backward()
            oDA.step()

            oDB.zero_grad()
            rB = DB(rb)
            fB = DB(fb.detach())
            lsDB = MS(rB, lr)
            lsDB = lsDB + MS(fB, lf)
            lsDB.backward()
            oDB.step()
            
            if it % 3 == 0:
                print("[%d %d]lsG: %.4f %.4f %.4f, lsDA: %.4f, lsDB: %.4f" % (ep, it, lsG1.item(), lsG2.item(), lsG3.item(), lsDA.item(), lsDB.item()))
        
        torch.save(GAB, './gab')
        torch.save(GBA, './gba')
        torch.save(DA, './da')
        torch.save(DB, './db')
        with torch.no_grad():
#             print(tnp.shape(dt[1][0:8].numpy()))
            oA = (np.vstack((ra[0:8].cpu(), fb[0:8].cpu(), rb[0:8].cpu(), fa[0:8].cpu())) * 127.5 + 127.5).astype('uint8')
#             print(np.shape(oA))
#             print(fb[-1])
#             print(oA[-1])
            ot = np.vstack(
                list(np.hstack(
                    list(oA[i].transpose(1, 2, 0) for i in range(j * 8, j * 8 + 8))
                ) for j in range(4))
            )
            ig = Image.fromarray(ot).convert('RGB')
            ig.save('%03d.jpg' % (ep))
            

if __name__ == '__main__':
    main()
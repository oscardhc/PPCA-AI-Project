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

from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
import tensorflow as tf
import tensorflow.summary as sm


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

        while sz > 64:
            model = model + [
                nn.Conv2d(ch, ch * 2, 4, 2, 1),
                nn.BatchNorm2d(ch * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            ch = ch * 2
            sz = sz // 2

        model = model + [
            nn.Conv2d(ch, 1, 4, 2, 1),
            #             nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    dv = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    nt = 7200
    ne = 200
    bs = 24

    #     GAB = Generator(256, 3).to(dv)
    #     GBA = Generator(256, 3).to(dv)
    #     DA = Discriminator(256, 3).to(dv)
    #     DB = Discriminator(256, 3).to(dv)
    #     GAB.apply(weights_init)
    #     GBA.apply(weights_init)
    #     DA.apply(weights_init)
    #     DB.apply(weights_init)
    GAB = torch.load('./gab').to(dv)
    GBA = torch.load('./gba').to(dv)
    DA = torch.load('./da').to(dv)
    DB = torch.load('./db').to(dv)
    wr = SummaryWriter('./run', flush_secs=3)

    ds = pixDataset('../PyTorch-GAN/data/ukiyoe2photo/train', nt)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
    ts = pixDataset('../PyTorch-GAN/data/ukiyoe2photo/test', 16)
    tl = torch.utils.data.DataLoader(ts, batch_size=16, shuffle=False, num_workers=4)

    MS = nn.MSELoss()
    L1 = nn.L1Loss()

    oG = optim.Adam(itertools.chain(GAB.parameters(), GBA.parameters()), lr=2e-4, betas=(0.5, 0.999))
    oDA = optim.Adam(DA.parameters(), lr=3e-4, betas=(0.5, 0.999))
    oDB = optim.Adam(DB.parameters(), lr=3e-4, betas=(0.5, 0.999))

    lr = torch.Tensor(np.zeros((bs, 1, 32, 32))).to(dv).fill_(1)
    lf = torch.Tensor(np.zeros((bs, 1, 32, 32))).to(dv).fill_(0)
    se = 188
    st = se * 100

    for _, im in enumerate(tl):
        break

    for ep in range(se, ne):
        ba = tqdm(total=len(dl.dataset))
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

            lsG = lsG1 * 0.5 + lsG2 * 2.0 + lsG3 * 5.0
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

            ba.update(dl.batch_size)

            if it % 3 == 0:
                st = st + 1
                ba.set_description("[%d %d]lsG: %.4f %.4f %.4f, lsDA: %.4f, lsDB: %.4f" % (
                ep, st, lsG1.item(), lsG2.item(), lsG3.item(), lsDA.item(), lsDB.item()))

                wr.add_scalar('scalar/lsD_A', lsDA, st)
                wr.add_scalar('scalar/lsD_B', lsDB, st)
                wr.add_scalar('scalar/lsG_1', lsG1, st)
                wr.add_scalar('scalar/lsG_2', lsG2, st)
                wr.add_scalar('scalar/lsG_3', lsG3, st)

        torch.save(GAB, './gab')
        torch.save(GBA, './gba')
        torch.save(DA, './da')
        torch.save(DB, './db')
        with torch.no_grad():
            oA = (GBA(im[1].float().to(dv)).cpu().numpy() * 255.0).astype('uint8')
            oA = np.vstack((oA, (GAB(im[0].float().to(dv)).cpu().numpy() * 255.0).astype('uint8')))
            ot = np.vstack(
                list(np.hstack(
                    list(oA[i].transpose(1, 2, 0) for i in range(j * 8, j * 8 + 8))
                ) for j in range(4))
            )
            ig = Image.fromarray(ot).convert('RGB')
            ig.save('./res/%03d.jpg' % (ep))
            #             print('!!!!!!!', np.shape(ot))
            mg = tf.compat.v1.summary.image('image/res%03d' % ep, ot.reshape((1, 1024, 2048, 3)))
            with tf.compat.v1.Session() as sess:
                # Run
                summary = sess.run(mg)
                # Write summary
                writer = tf.compat.v1.summary.FileWriter('run')
                writer.add_summary(summary)
                writer.close()

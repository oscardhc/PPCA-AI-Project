from __future__ import print_function, division
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from AlphabetDataset import Alphabet
from PIL import Image
from tqdm import tqdm


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        channel = [1024, 512, 256, 128, 32, 3]
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, channel[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(True),
            nn.ConvTranspose2d(channel[0], channel[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(channel[1], channel[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU(True),
            nn.ConvTranspose2d(channel[2], channel[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel[3]),
            nn.ReLU(True),
            nn.ConvTranspose2d(channel[3], channel[4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel[4]),
            nn.ReLU(True),
            nn.ConvTranspose2d(channel[4], channel[5], 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        channel = [3, 32, 128, 256, 512, 1024]
        self.main = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel[1], channel[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel[2], channel[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel[3], channel[4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel[4], channel[5], 4, 2, 1, bias=False),
            nn.BatchNorm2d(channel[5]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel[5], 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    bs = 125
    dataset = Alphabet(12500)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)
    criterion = nn.BCELoss()

    #     G = Generator().to(device)
    #     D = Discriminator().to(device)
    #     G.apply(weights_init)
    #     D.apply(weights_init)
    G = torch.load('./g.t7').to(device)
    D = torch.load('./d.t7').to(device)

    optG = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))

    #     fixedNoise = torch.randn(bs, 100, 1, 1, device=device)
    #     np.save('noise.npy', fixedNoise.detach().cpu())
    fixedNoise = torch.tensor(np.load("noise.npy")).to(device)

    realLabel = 1
    fakeLabel = 0

    imgList = []
    GLosses = []
    DLosses = []
    iters = 0
    numE = 900

    for epoch in range(421, numE + 1):
        progress = tqdm(total=len(dataloader.dataset))
        for i, data in enumerate(dataloader, 0):

            for j in range(1):
                D.zero_grad()
                real = data[0].float().to(device)
                bn = real.size(0)
                #                 label = torch.full((bn, ), realLabel, device=device)
                label = data[1].float().to(device)

                out = D(real).view(-1)
                errDreal = criterion(out, label)
                errDreal.backward()
                Dx = out.mean().item()

                noise = torch.randn(bs, 100, 1, 1, device=device)
                fake = G(noise)
                label.fill_(fakeLabel)

                out = D(fake.detach()).view(-1)
                errDfake = criterion(out, label)
                errDfake.backward()
                DGz1 = out.mean().item()

                errD = errDreal + errDfake
                optD.step()

            G.zero_grad()
            label.fill_(realLabel)

            out = D(fake).view(-1)
            errG = criterion(out, label)
            errG.backward()
            DGz2 = out.mean().item()
            optG.step()

            progress.update(dataloader.batch_size)
            progress.set_description('[%d/%d] Loss_D: %.4f Loss_G: %.4f'
                                     % (epoch, numE,
                                        errD.item(), errG.item()))

            GLosses.append(errG.item())
            DLosses.append(errD.item())

        if epoch % 3 == 0:
            torch.save(G, os.path.join(os.getcwd(), "g.t7"))
            torch.save(D, os.path.join(os.getcwd(), "d.t7"))
            with torch.no_grad():
                fake = G(fixedNoise).detach().cpu()
                imgList.append(vutils.make_grid(fake, padding=2, normalize=True))
                rr = np.ndarray((1))
                for I in range(8):
                    ar = np.ndarray((1))
                    for J in range(8):
                        if np.size(ar) == 1:
                            ar = (fake[I * 8 + J].numpy() * 255.0).astype('uint8').transpose(1, 2, 0)
                        else:
                            ar = np.hstack((ar, (fake[I * 8 + J].numpy() * 255.0).astype('uint8').transpose(1, 2, 0)))
                    if np.size(rr) == 1:
                        rr = ar
                    else:
                        #                     print(np.shape(rr), np.shape(ar))
                        rr = np.vstack((rr, ar))
                img = Image.fromarray(rr).convert('RGB')
                img.save('./%03d.png' % (epoch))

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(GLosses, label="G")
    plt.plot(DLosses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('res.png')

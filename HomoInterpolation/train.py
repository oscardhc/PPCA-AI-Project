import torch
import torch.nn as nn
import torch.optim as optim
import model as m
from dataset import CelebADataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image


class Program(object):
    def __init__(self, imgsize, device, attr, toLoad):
        super().__init__()
        self.epoch = 100
        self.batch_size = 16
        self.feat_n = len(attr)
        self.batch_penalty = 1
        self.imgsize = imgsize

        self.fixedImgs = []
        self.total_step = 0

        self.E = m.Encoder().to(device)
        self.D = m.Decoder().to(device)
        self.dis = m.Discriminator(self.feat_n).to(device)
        self.I = m.Interp(self.feat_n).to(device)
        self.P = m.KG().to(device)
        self.Teacher = m.VGG().to(device)

        if toLoad:
            self.load_model()

        self.dataset = CelebADataset(192000, '/Users/oscar/Downloads/celeba-dataset', self.imgsize, attr)
        self.device = device

    def train(self):
        self._train(self.device)

    def _train(self, device):
        E_optim = optim.Adam(self.E.parameters(), lr=0.0002)
        D_optim = optim.Adam(self.D.parameters(), lr=0.0002)
        dis_optim = optim.Adam(self.dis.parameters(), lr=0.0002)
        I_optim = optim.Adam(self.I.parameters(), lr=0.0002)
        P_optim = optim.Adam(self.P.parameters(), lr=0.0002)

        MSE_criterion = nn.MSELoss().to(device)
        BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean').to(device)

        full_strenth = torch.ones(self.batch_size, self.feat_n).to(device)
        """load data there"""
        dataset = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        for i in range(self.epoch):
            for t, (images, attr) in enumerate(dataset):
                attr = attr.transpose(1, 0)

                images = images.float().to(device)
                attr = attr.float().to(device)

                strength = torch.randn(self.batch_size, self.feat_n).to(device)
                perm = torch.randperm(self.batch_size).to(device)

                real_F = self.E(images)
                perm_F = real_F[perm]
                interp_F = self.I(real_F, perm_F, strength)

                perm_attr = []
                interp_attr = []
                for att in attr:
                    perm_attr += [att[perm]]
                # ??
                for i, (att, perm_att) in enumerate(zip(attr, perm_attr)):
                    interp_attr += [att + strength[:, i:i + 1] * (perm_att - att)]
                # ??

                E_optim.zero_grad()
                D_optim.zero_grad()
                dis_optim.zero_grad()
                I_optim.zero_grad()
                P_optim.zero_grad()

                real_dec = self.D(real_F)

                print(real_dec.size(), images.size())
                D_loss = MSE_criterion(real_dec, images)
                """another part of loss relys on the VGG network"""
                D_loss.backward(retain_graph=True)
                D_optim.step()

                E_optim.zero_grad()
                D_optim.zero_grad()

                real_critc, real_attr = self.dis(real_F.detach())
                interp_critic, interp_homo_attr = self.dis(interp_F.detach())
                dis_loss = (interp_critic - real_critc).mean()
                """calculate the gradient penalty there"""
                cl_loss = 0
                tmp_real_attr = [att.detach() for att in real_attr]
                for real_att, interp_homo_att in zip(tmp_real_attr, interp_homo_attr):
                    cl_loss += BCE_criterion(interp_homo_att, real_att) / real_att.size(0)
                dis_loss += cl_loss
                """calculate the classfication loss there. done"""
                dis_loss.backward(retain_graph=True)
                dis_optim.step()

                E_optim.zero_grad()
                D_optim.zero_grad()
                dis_optim.zero_grad()
                I_optim.zero_grad()
                P_optim.zero_grad()

                vgg_feat = self.Teacher(images).detach()
                vgg_loss = MSE_criterion(vgg_feat, self.P(real_F.detach()))
                vgg_loss.backward()
                P_optim.step()
                """calculate the KG loss by using VGG as the teacher here and modify the network.done"""

                if t % self.batch_penalty == 0:
                    E_optim.zero_grad()
                    D_optim.zero_grad()
                    dis_optim.zero_grad()
                    I_optim.zero_grad()
                    P_optim.zero_grad()

                    interp_critic, interp_homo_attr = self.dis(interp_F)
                    EI_loss = -interp_critic.mean()
                    cl_loss = 0
                    tmp_interp_attr = [att.detach() for att in interp_attr]
                    for interp_att, homo_att in zip(tmp_interp_attr, interp_homo_attr):
                        cl_loss += BCE_criterion(homo_att, interp_att)
                    cl_loss /= tmp_interp_attr.size(0)
                    EI_loss += cl_loss
                    """calculate the classfication loss here. done"""
                    total_interp_F = self.I(real_F, perm_F, full_strenth)
                    EI_loss += MSE_criterion(total_interp_F, perm_F)
                    EI_loss.backward()

                    E_optim.step()
                    I_optim.step()

                self.total_step += 1

                if self.total_step % 1000 == 0:
                    self.save_model()
                    """out put some information about the status there"""

                if self.total_step % 500 == 0:
                    self.showResult()
                    """test the result of the net there"""

    def save_model(self):
        with open('mata.txt', 'w') as f:
            print(self.total_step, file=f)
        torch.save(self.E.state_dict().cpu(), "encoder.pth")
        torch.save(self.D.state_dict().cpu(), "decoder.pth")
        torch.save(self.dis.state_dict().cpu(), "Discriminator.pth")
        torch.save(self.I.state_dict().cpu(), "Interp.pth")
        torch.save(self.P.state_dict().cpu(), "KG.pth")

    def load_model(self):
        with open('mata.txt', 'r') as f:
            ar = f.read().split(' ')
            self.total_step = int(ar[0])
        self.E.load_state_dict(torch.load("encoder.pth"))
        self.D.load_state_dict(torch.load("decoder.pth"))
        self.dis.load_state_dict(torch.load("Discriminator.pth"))
        self.I.load_state_dict(torch.load("Interp.pth"))
        self.P.load_state_dict(torch.load("KG.pth"))

    def run(self, imageA, imageB, strength):
        """batch size!"""
        imageA = imageA.reshape((1, imageA.size(0), imageA.size(1), imageA.size(2)))
        imageB = imageB.reshape((1, imageA.size(0), imageA.size(1), imageA.size(2)))
        featA = self.E(imageA)
        featB = self.E(imageB)
        feat_interp = self.I(featA, featB, strength)
        return self.D(feat_interp)

    def showResult(self):
        rg = len(self.fixedImgs) - 1
        tt = []
        for i in range(rg):
            for j in range(self.feat_n):
                str = torch.zeros((1, self.feat_n))
                tmp = []
                for k in range(0, 1.5, 0.3):
                    str[0][j] = k
                    res = self.run(self.fixedImgs[i], self.fixedImgs[i + 1], str)
                    res.squeeze_()
                    res = res.transpose(1, 2, 0)
                    tmp.append(res)
                tt.append(np.hstack(tmp))
        ary = np.vstack(tt)
        img = Image.fromarray(ary)
        img.save('res-%06d.jpg' % (self.total_step))

import torch
import torch.nn as nn
import torch.optim as optim
import model as m
from dataset import CelebADataset
from torch.utils.data import DataLoader


class Program(object):
    def __init__(self, device):
        super().__init__()
        self.epoch = 100
        self.batch_size = 16
        self.feat_n = 10
        self.batch_penalty = 5

        self.E = m.Encoder().to(device)
        self.D = m.Decoder().to(device)
        self.dis = m.Discriminator().to(device)
        self.I = m.Interp().to(device)
        self.P = m.KG().to(device)
        self.Teacher = m.VGG().to(device)

    def train(self, device):
        E_optim = optim.Adam(self.E.parameters(), lr=0.0002).to(device)
        D_optim = optim.Adam(self.D.parameters(), lr=0.0002).to(device)
        dis_optim = optim.Adam(self.dis.parameters(), lr=0.0002).to(device)
        I_optim = optim.Adam(self.I.parameters(), lr=0.0002).to(device)
        P_optim = optim.Adam(self.P.parameters(), lr=0.0002).to(device)


        MSE_criterion = nn.MSELoss().to(device)
        BCE_criterion = nn.BCEWithLogitsLoss().to(device)

        full_strenth = torch.ones(self.batch_size, self.feat_n).to(device)
        """load data there"""
        dataset = DataLoader(CelebADataset(), batch_size=self.batch_size, shuffle=True, num_workers=4)

        for i in range(self.epoch):
            for t, images, attr in enumerate(dataset):
                attr = attr.transpose(1, 0)

                images = images.to(device)
                attr = attr.to(device)

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
                    cl_loss += BCE_criterion(interp_homo_att, real_att, size_average=True)
                cl_loss /= tmp_real_attr.size(0)
                dis_loss += cl_loss
                """calculate the classfication loss there. done"""
                dis_loss.backward(retain_grapth=True)
                dis_optim.step()

                E_optim.zero_grad()
                D_optim.zero_grad()
                dis_optim.zero_grad()
                I_optim.zero_grad()
                P_optim.zero_grad()

                vgg_feat = self.Teacher(images).detach()
                vgg_loss = MSE_criterion(vgg_feat, P(real_F.detach()))
                vgg_loss.backward()
                self.P.step()
                """calculate the KG loss by using VGG as the teacher here and modify the network.done"""

                if t % self.batch_penalty == 0:
                    E_optim.zero_grad()
                    D_optim.zero_grad()
                    dis_optim.zero_grad()
                    I_optim.zero_grad()
                    P_optim.zero_grad()

                    interp_critic, interp_homo_attr = dis(interp_F)
                    EI_loss = -interp_critic.mean()
                    cl_loss = 0
                    tmp_interp_attr = [att.detach() for att in interp_attr]
                    for interp_att, homo_att in zip(tmp_interp_attr, interp_homo_attr):
                        cl_loss += BCE_criterion(homo_att, interp_att, size_averate=True)
                    cl_loss /= tmp_interp_attr.size(0)
                    EI_loss += cl_loss
                    """calculate the classfication loss here. done"""
                    total_interp_F = self.I(real_F, perm_F, full_strenth)
                    EI_loss += MSE_criterion(total_interp_F, perm_F)
                    EI_loss.backward()

                    E_optim.step()
                    I_optim.step()

                if t % 1000 == 0:
                    torch.save(self.E.state_dict().cpu(), "encoder.pth")
                    torch.save(self.D.state_dict().cpu(), "decoder.pth")
                    torch.save(self.dis.state_dict().cpu(), "Discriminator.pth")
                    torch.save(self.I.state_dict().cpu(), "Interp.pth")
                    torch.save(self.P.state_dict().cpu(), "KG.pth")
                    """out put some information about the status there"""
            """test the result of the net there"""

    def run(self, imageA, imageB, strength):
        featA = self.E(imageA)
        featB = self.E(imageB)
        feat_interp = self.I(imageA, imageB, strength)
        return self.D(image_interp)
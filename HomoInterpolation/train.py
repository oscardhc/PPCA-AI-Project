import torch
import torch.nn as nn
import torch.optim as optim
import model as m
from dataset import CelebADataset
from torch.utils.data import DataLoader


def train(device):

    epoch = 100
    batch_size = 16
    feat_n = 10
    batch_penalty = 5

    E = m.Encoder().to(device)
    D = m.Decoder().to(device)
    dis = m.Discriminator().to(device)
    I = m.Interp().to(device)
    P = m.KG().to(device)

    E_optim = optim.Adam(E.parameters(), lr=0.0002)
    D_optim = optim.Adam(D.parameters(), lr=0.0002)
    dis_optim = optim.Adam(dis.parameters(), lr=0.0002)
    I_optim = optim.Adam(I.parameters(), lr=0.0002)
    P_optim = optim.Adam(P.parameters(), lr=0.0002)

    MSE_criterion = nn.MSELoss().to(device)
    BCE_criterion = nn.BCEWithLogitsLoss().to(device)

    full_strenth = torch.ones(batch_size, feat_n).to(device)
    """load data there"""
    dataset = DataLoader(CelebADataset(), batch_size=batch_size, shuffle=True, num_workers=4)

    for i in range(epoch):
        for t, images, attr in enumerate(dataset):

            images = images.to(device)
            attr = attr.to(device)

            strength = torch.randn(batch_size, feat_n).to(device)
            perm = torch.randperm(batch_size).to(device)

            real_F = E(images)
            perm_F = real_F[perm]
            interp_F = I(real_F, perm_F, strength)

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

            real_dec = D(real_F)
            D_loss = MSE_criterion(real_dec, images)
            """another part of loss relys on the VGG network"""
            D_loss.backward(retain_graph=True)
            D_optim.step()

            E_optim.zero_grad()
            D_optim.zero_grad()

            real_critc, real_attr = dis(real_F.detach())
            interp_critic, interp_homo_attr = dis(interp_F.detach())
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

            """calculate the KG loss by using VGG as the teacher here and modify the network"""

            if t % batch_penalty == 0:
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
                total_interp_F = I(real_F, perm_F, full_strenth)
                EI_loss += MSE_criterion(total_interp_F, perm_F)
                EI_loss.backward()

                E_optim.step()
                I_optim.step()

            if t % 1000 == 0:
                torch.save(E.state_dict(), "encoder.pth")
                torch.save(D.state_dict(), "decoder.pth")
                torch.save(dis.state_dict(), "Discriminator.pth")
                torch.save(I.state_dict(), "Interp.pth")
                torch.save(P.state_dict(), "KG.pth")
                """out put some information about the status there"""
        """test the result of the net there"""
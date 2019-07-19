import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import model as m


epoch = 100
batch_size = 16
feat_n = 10
batch_penalty = 5

if __name__ == "__main__":
    CEL_criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    E = m.Encoder()
    D = m.Decoder()
    dis = m.Discriminator()
    I = m.Interp()
    P = m.KG()

    E_optim = optim.Adam(E.parameters(), lr=0.0002)
    D_optim = optim.Adam(D.parameters(), lr=0.0002)
    dis_optim = optim.Adam(dis.parameters(), lr=0.0002)
    I_optim = optim.Adam(I.parameters(), lr=0.0002)
    P_optim = optim.Adam(P.parameters(), lr=0.0002)

    MSE_criterion = nn.MSELoss()
    BCE_criterion = nn.BCELoss()

    full_strenth = torch.tensor(torch.ones(batch_size, feat_n))
    """load data there"""
    dataset = 0

    for i in range(epoch):
        for t, images, attr in enumerate(dataset):
            
            strength = torch.tensor(torch.randn(batch_size, feat_n))
            perm = torch.randperm(batch_size)

            real_F = E(images)
            perm_F = real_F[perm]
            interp_F = I(real_F, perm_F, strength)

            perm_attr = []
            interp_attr = []
            for att in attr:
                perm_attr += [att[perm]]
            # ??
            for i, (att, perm_att) in enumerate(zip(attr, perm_attr)):
                interp_attr += [att + strength[i] * (perm_att - att)]
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
            interp_critic, interp_attr = dis(interp_F.detach())
            dis_loss = (interp_critic - real_critc).mean()
            """calculate the gradient penalty there"""
            """calculate the classfication loss there"""
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

                interp_critic, interp_attr = dis(interp_F)
                EI_loss = -interp_critic.mean()
                """calculate the classfication loss here"""
                total_interp_F = I(real_F, perm_F, full_strenth)
                EI_loss += MSE_criterion(total_interp_F, perm_F)
                EI_loss.backward()

                E_optim.step()
                I_optim.step()





        torch.save(E.state_dict(), "encoder.pth")
        torch.save(D.state_dict(), "decoder.pth")
        torch.save(dis.state_dict(), "Discriminator.pth")
        torch.save(I.state_dict(), "Interp.pth")
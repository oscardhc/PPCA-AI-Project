import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import model as m

epoch = 100
batch_size = 16
batch_penalty = 5
def grad_loss(matrix):
    return torch.sum(matrix[:, :, :, :-1] - matrix[:, :, :, 1:]) + \
    torch.sum(matrix[:, :, :-1, :] - matrix[:, :, 1:, :])

def train(device):
    G = m.generator()       # color, attention
    D = m.discriminator()   # I, y
    G = G.to(device)
    D = D.to(device)

    G_optim = optim.Adam(G.parameters(), lr=0.0002)
    D_optim = optim.Adam(D.parameters(), lr=0.0002)

    MSE_criterion = nn.MSELoss().to(device)
    L1_criterion = nn.L1Loss().to(device)
    """dataset there"""
    dataset = []

    for _ in range(epoch):
        for i, images, tags in enumerate(dataset):
            images = images.to(device)
            tags = tags.to(device)
            perm = torch.randperm(batch_size)
            f_tags = tags[perm]

            A_in, C_in = G(images, f_tags)
            full_in = torch.ones_like(A_in, device=device)
            f_images = (full_in - A_in).mul(C_in) + A_in.mul(images)
            """or, instead of mul, it is mm? since it simply uses operator*"""

            A_f, C_f = G(f_images)
            full_f = torch.ones_like(A_f)
            back_images = (full_f - A_f).mul(C_f) + A_in.mul(f_images)

            DIg, DYg = D(f_images)
            DIi, DYi = D(images)

            Li = DIg.mean() - DIi.mean()
            """add gradient penalty there"""
            La = grad_loss(A_in) + grad_loss(A_f) + A_in.mean() + A_f.mean()
            Lcon = MSE_criterion(DYg, f_tags) + MSE_criterion(DYi, tags)
            Lcon /= batch_size
            Lcyc = L1_criterion(back_images, images)

            G_optim.zero_grad()
            D_optim.zero_grad()
            loss = Li + La + Lcon + Lcyc
            loss.backward()
            G_optim.step()
            D_optim.step()
            """maybe a const to modify the training penalty of G is needed there"""

            if i % 1000 == 0:
                torch.save(G.state_dict().cpu(), "gen.pth")
                torch.save(D.state_dict().cpu(), "dis.pth")
                # and output the loss now there?
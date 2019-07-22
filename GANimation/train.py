import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import model as m


class Program(object):
    def __init__(self, device):
        super().__init__()
        self.epoch = 100
        self.batch_size = 16
        self.batch_penalty = 5

        G = m.generator()       # color, attention
        D = m.discriminator()   # I, y
        self.G = G.to(device)
        self.D = D.to(device)
    def grad_loss(self, matrix):
        return torch.sum(matrix[:, :, :, :-1] - matrix[:, :, :, 1:]) + \
        torch.sum(matrix[:, :, :-1, :] - matrix[:, :, 1:, :])

    def train(self, device):
        G_optim = optim.Adam(self.G.parameters(), lr=0.0002)
        D_optim = optim.Adam(self.D.parameters(), lr=0.0002)

        MSE_criterion = nn.MSELoss().to(device)
        L1_criterion = nn.L1Loss().to(device)
        """dataset here"""
        dataset = []

        for _ in range(epoch):
            for i, images, tags in enumerate(dataset):
                b_size = images.siez(0)
                images = images.to(device)
                tags = tags.to(device)
                perm = torch.randperm(b_size)
                f_tags = tags[perm]

                A_in, C_in = self.G(images, f_tags)
                full_in = torch.ones_like(A_in, device=device)
                f_images = (full_in - A_in) * C_in + A_in * images
                """or, instead of mul, it is mm? since it simply uses operator*"""

                A_f, C_f = self.G(f_images)
                full_f = torch.ones_like(A_f)
                back_images = (full_f - A_f) * C_f + A_in * f_images

                DIg, DYg = self.D(f_images)
                DIi, DYi = self.D(images)

                Li = DIg.mean() - DIi.mean()
                """add gradient penalty there"""
                La = grad_loss(A_in) + grad_loss(A_f) + A_in.mean() + A_f.mean()
                Lcon = MSE_criterion(DYg, f_tags) + MSE_criterion(DYi, tags)
                Lcon /= b_size
                Lcyc = L1_criterion(back_images, images)

                G_optim.zero_grad()
                D_optim.zero_grad()
                loss = Li + La + Lcon + Lcyc
                loss.backward()
                G_optim.step()
                D_optim.step()
                """maybe a const to modify the training penalty of G is needed there"""

                if i % 1000 == 0:
                    save_model()
                    # and output the loss now there?
    
    def save_model(self):
        torch.save(self.G.state_dict().cpu(), "gen.pth")
        torch.save(self.D.state_dict().cpu(), "dis.pth")
    def load_model(self):
        self.G.load_state_dice(torch.load("gen.pth"))
        self.D.load_state_dice(torch.load("dis.pth"))
    def run(self, image, f_tags, strenth):
        A, C = self.G(image, f_tags)
        f_images = (strenth - A) * C + A * image
        return f_images
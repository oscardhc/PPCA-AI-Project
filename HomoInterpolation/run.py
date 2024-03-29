import torch
import torch.nn as nn
import torch.optim as optim
from dataset import *
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorboardX
from model import *
import json


class Program(object):

    def __init__(self, imgsize, device, attr, toLoad, batchsize, epo, ste, batchpen, path, tpath, mpath, vpath, ppath):
        super().__init__()

        self.tmppath = ppath
        self.epoch = epo
        
        self.total_step = ste
        self.batch_size = batchsize

        self.attrName = attr
        self.attr = []

        for group in self.attrName:
            self.attr.append(len(group))

        self.attr_n = len(attr)
        self.attrname = attr

        self.batch_penalty = batchpen
        self.imgsize = imgsize

        self.datapath = path
        self.device = device

        self.testpath = tpath

        self.vggpath = vpath
        self.E = Encoder(self.vggpath).to( device)
        self.D = Decoder().to(device)
        self.dis = Discriminator(self.attr).to(device)
        # mention that the attr is an 1-dim numpy
        self.I = Interp(self.attr_n + 1).to(device)
        self.P = KG().to(device)
        self.Teacher = VGG(self.vggpath).to(device)
        self.imdis = Image_Dis().to(device)

        self.modelpath = mpath

        if toLoad:
            self.load_model()

    def train(self):
        self._train(self.device)

    def _train(self, device):

        wr = tensorboardX.SummaryWriter('./log', flush_secs=2)
        
        self.dataset = CelebADataset(192000, self.datapath, self.imgsize, self.attrName)
        self.fixedfix = getTestImages(os.path.join(self.testpath), self.imgsize)
        self.fixedlen = len(self.fixedfix)

        E_optim = optim.Adam(self.E.parameters(), lr=0.0001)
        D_optim = optim.Adam(self.D.parameters(), lr=0.0001)
        dis_optim = optim.Adam(self.dis.parameters(), lr=0.0001)
        I_optim = optim.Adam(self.I.parameters(), lr=0.0001)
        P_optim = optim.Adam(self.P.parameters(), lr=0.0001)
        Im_dis_optim = optim.Adam(self.P.parameters(), lr=0.0002)
        # cautious! betas=(0.5, 0.999) can easily lead to overfitting

        MSE_criterion = nn.MSELoss().to(device)
        BCE_criterion = nn.BCEWithLogitsLoss().to(device)

        full_strenth = torch.ones(self.batch_size, self.attr_n + 1).to(device)
        """load data there"""
        dataset = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        for ep in range(self.epoch):
            process = tqdm(total=(len(dataset)))
            for t, (images, attr) in enumerate(dataset):            
                images = images.float().to(device)
                attr = [aa.to(device) for aa in attr]

                strength = torch.rand(images.size(0), self.attr_n + 1).to(device)
                perm = torch.randperm(images.size(0)).to(device)

                real_F = self.E(images)
                perm_F = real_F[perm]
                interp_F = self.I(real_F, perm_F, strength)
                with torch.no_grad():
#                     wr.add_graph(self.E, images, verbose=True)
                    wr.add_graph(self.I, (real_F, perm_F, strength), verbosed=True)

                perm_attr = []
                interp_attr = []
                for att in attr:
                    perm_attr += [att[perm]]
                for i, (att, perm_att) in enumerate(zip(attr, perm_attr)):
                    interp_attr += [att + strength[:, i:i + 1] * (perm_att - att)]
                """run the network above"""
                E_optim.zero_grad()
                D_optim.zero_grad()
                dis_optim.zero_grad()
                I_optim.zero_grad()
                P_optim.zero_grad()

                real_dec = self.D(real_F.detach())

                D_loss = MSE_criterion(real_dec, images.detach())
                """calculate the reconstruction loss above"""
                dgg_loss = MSE_criterion(self.Teacher(real_dec), self.Teacher(images))
                D_loss += dgg_loss
                """calculate the perceptural loss above"""
                D_loss.backward(retain_graph=True)
                D_optim.step()

                E_optim.zero_grad()
                D_optim.zero_grad()
                dis_optim.zero_grad()
                I_optim.zero_grad()
                P_optim.zero_grad()

                real_critc, real_attr = self.dis(real_F.detach())
                interp_critic, interp_homo_attr = self.dis(interp_F.detach())
                dis_loss = (interp_critic - real_critc).mean()
                """calculate the WGAN loss above"""
                alpha = torch.tensor(np.random.random((real_critc.size(0), 1, 1, 1))).float().to(device)
                mid_F = real_F + alpha * (interp_F - real_F)
                mid_F.requires_grad_(True)
                mid_critic, _ = self.dis(mid_F)
                grad = torch.autograd.grad(
                    outputs=mid_critic,
                    inputs=mid_F,
                    grad_outputs=torch.ones(mid_critic.size()).to(self.device),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0].view(real_critc.size(0), -1)
                gp_loss = 100 * ((grad.norm(2, dim=1) - 1) ** 2).mean()
                dis_loss += gp_loss
                """calculate the gradient penalty above"""
                cl_loss = 0
                tmp_real_attr = [att.detach() for att in attr]
                for real_att, interp_homo_att in zip(tmp_real_attr, interp_homo_attr):
                    cl_loss += BCE_criterion(interp_homo_att, real_att) / (real_att.size(0))
                dis_loss += cl_loss
                """calculate the classification loss above"""
                dis_loss.backward(retain_graph=True)
                dis_optim.step()

                E_optim.zero_grad()
                D_optim.zero_grad()
                dis_optim.zero_grad()
                I_optim.zero_grad()
                P_optim.zero_grad()

                outimg = self.D(interp_F)
                imgJudge = self.imdis(outimg)
                imgJudge_dis = self.imdis(outimg.detach())
                isreal = torch.ones_like(imgJudge).to(device)
                isfalse = torch.zeros_like(imgJudge).to(device)
                realImgJudge = self.imdis(images)

                img_real_loss = MSE_criterion(imgJudge_dis, isfalse).mean()
                img_real_loss.backward(retain_graph=True)
                img_rr_loss = MSE_criterion(realImgJudge, isreal).mean()
                img_rr_loss.backward(retain_graph=True)
                Im_dis_optim.step()

                E_optim.zero_grad()
                D_optim.zero_grad()
                dis_optim.zero_grad()
                I_optim.zero_grad()
                P_optim.zero_grad()

                im_false_loss = MSE_criterion(imgJudge, isreal).mean()
                img_real_loss.backward()
                if t % self.batch_penalty == 0:
                    I_optim.step()
                    E_optim.step()
                D_optim.step()

                E_optim.zero_grad()
                D_optim.zero_grad()
                dis_optim.zero_grad()
                I_optim.zero_grad()
                P_optim.zero_grad()

                vgg_feat = self.Teacher(images).detach()
                vgg_loss = MSE_criterion(vgg_feat, self.P(real_F.detach()))
                vgg_loss.backward()
                P_optim.step()
                """calculate the KG loss above"""

                if t % self.batch_penalty == 0:
                    E_optim.zero_grad()
                    D_optim.zero_grad()
                    dis_optim.zero_grad()
                    I_optim.zero_grad()
                    P_optim.zero_grad()

                    interp_critic, interp_homo_attr = self.dis(interp_F)
                    EI_loss = -interp_critic.mean()
                    """calculate the WGAN loss above"""
                    cl_loss = 0
                    tmp_interp_attr = [att.detach() for att in interp_attr]
                    for interp_att, homo_att in zip(tmp_interp_attr, interp_homo_attr):
                        cl_loss += BCE_criterion(homo_att, interp_att) / (homo_att.size(0))
                    EI_loss += cl_loss
                    """calculate the classification loss above"""
                    total_interp_F = self.I(real_F.detach(), perm_F.detach(), full_strenth)
                    EI_loss += MSE_criterion(total_interp_F, perm_F.detach())
                    """calculate the interpolation loss above"""
                    EI_real_dec = self.D(real_F)
                    EI_loss += MSE_criterion(EI_real_dec, images.detach())
                    """calculate the reconstruction loss above"""
                    EI_vgg_feat = self.Teacher(images).detach()
                    # this one is supposed to be only pass throw first few layers instead of the whole VGG, 
                    # otherwise, it is easy to cause overfitting.
                    EI_vgg_loss = MSE_criterion(EI_vgg_feat, self.P(real_F))
                    EI_loss += EI_vgg_loss
                    """calculate the homomorphic gap above"""
                    EI_loss += MSE_criterion(self.Teacher(EI_real_dec), self.Teacher(images.detach()))
                    """calculate the KG loss above"""
                    EI_loss.backward()

                    E_optim.step()
                    I_optim.step()

                if self.total_step % 100 == 0:
                    self.showResult()
                    """test the result of the net there"""

                if self.total_step % 1000 == 0:
                    self.save_model()
                    """out put some information about the status there"""

                self.total_step += 1
                process.update(1)
                process.set_description('[%d] D:%.3f DIS:%.3f VGG:%.3f EI:%.3f' % (
                    self.total_step, D_loss.item(), dis_loss.item(), vgg_loss.item(), EI_loss.item()))

                if self.total_step % 10 == 0:
                    with open(self.tmppath + 'train/output.json', "w") as f:
                        output_json = {
                            'output image': self.tmppath + 'train/res.jpg',
                            'model path': self.modelpath,
                            'total step': self.total_step,
                            'D Loss': D_loss.item(),
                            'DIS Loss': dis_loss.item(),
                            'VGG Loss': vgg_loss.item(),
                            'EI Loss': EI_loss.item()
                        }
                        json.dump(output_json, f)
                        f.flush() # 中间结果更新时需要flush
                
                
                wr.add_scalar('scalar/D', D_loss.item(), self.total_step)
                wr.add_scalar('scalar/DIS', dis_loss.item(), self.total_step)
                wr.add_scalar('scalar/VGG', vgg_loss.item(), self.total_step)
                wr.add_scalar('scalar/EI', EI_loss.item(), self.total_step)

    def save_model(self):
        print('saving model...')
        with open(self.modelpath + "mata.txt", "w") as f:
            print(self.total_step, file=f)
        torch.save(self.E.state_dict(), self.modelpath + "encoder.pth")
        torch.save(self.D.state_dict(), self.modelpath + "decoder.pth")
        torch.save(self.dis.state_dict(), self.modelpath + "Discriminator.pth")
        torch.save(self.I.state_dict(), self.modelpath + "Interp.pth")
        torch.save(self.P.state_dict(), self.modelpath + "KG.pth")

    def load_model(self):
        print('loading model...')
        self.E.load_state_dict(torch.load(self.modelpath + "encoder.pth"))
        self.D.load_state_dict(torch.load(self.modelpath + "decoder.pth"))
        self.dis.load_state_dict(torch.load(self.modelpath + "Discriminator.pth"))
        self.I.load_state_dict(torch.load(self.modelpath + "Interp.pth"))
        self.P.load_state_dict(torch.load(self.modelpath + "KG.pth"))

    def run(self, imageA, imageB, strength, flag=True):
        imageA = imageA.reshape((1, 3, self.imgsize, self.imgsize))
        imageB = imageB.reshape((1, 3, self.imgsize, self.imgsize))
        featA = self.E(imageA)
        featB = self.E(imageB)
        if flag:
            feat_interp = self.I(featA, featB, strength)
        else:
            feat_interp = featA
        return self.D(feat_interp)

    def showArray(self, arr):
        tt = []
        rg = len(arr) - 1
        with torch.no_grad():
            for i in range(rg):
                ste = torch.zeros(1, self.attr_n + 1).to(self.device)
                tmp = []
                tmp.append(arr[i].transpose(1, 2, 0))
                for j in range(self.attr_n + 1):
                    for k in [1.0]:
                        ste[0][j] = k
                        res = self.run(torch.tensor(arr[i]).float().to(self.device),
                                       torch.tensor(arr[i + 1]).float().to(self.device),
                                       ste)
                        res.squeeze_()
                        res = res.detach().cpu().numpy().transpose(1, 2, 0)
                        tmp.append(res)
                tmp.append(arr[i + 1].transpose(1, 2, 0))
                tt.append(np.hstack(tmp))
        return tt

    def showResult(self):
        tt = []
        tt += self.showArray(self.fixedfix)
        ary = np.vstack(tt)
        img = Image.fromarray((ary * 255).astype('uint8'))
        img.save(self.tmppath + 'train/res.jpg')
        
    def predict(self, pa, pb, ste):
        im = getTestPair(pa, pb)
        res = self.run(torch.tensor(im[0]).float().to(self.device), torch.tensor(im[1]).float().to(self.device), torch.tensor([ste]).float().to(self.device))
        res.squeeze_()
        res = res.detach().cpu().numpy().transpose(1, 2, 0)
        img = Image.fromarray((res * 255).astype('uint8'))
        return img

import glob
import random
import os

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class pixDataset(torch.utils.data.Dataset):
    def __init__(self, path, num):
        self.fileA = sorted(glob.glob(path + "/A/" + "/*.*"))
        self.fileB = sorted(glob.glob(path + "/B/" + "/*.*"))
        self.num = num
        self.path = path

    def __len__(self):
        return self.num

    def getImage(self, path):
        img = Image.open(path)
        img = img.resize((256, 256))
        img.convert('RGB')
        ret = []
        try:
            ret = (np.array(img) / 255.0).transpose(2, 0, 1)
        except:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(img.mode, np.shape(np.array(img)))
            print(path)

        #         print('?????????', np.shape(ret), path)
        return ret

    def __getitem__(self, index):
        ima = self.fileA[index % len(self.fileA)]
        imb = self.fileB[index % len(self.fileB)]
        return (self.getImage(ima), self.getImage(imb))

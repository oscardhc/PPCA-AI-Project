
import glob
import random
import os

import torch
from PIL import Image
import numpy as np
import csv
import torch.utils.data

class CelebADataset(torch.utils.data.Dataset):

    def __init__(self, num, path, picSize, attr):
        super(CelebADataset, self).__init__()
        self.path = path
        self.num = num
        self.picSize = picSize
        self.name = []
        self.attr = []
        self.orit = []
        
        with open(path + '/celeba-with-orientation.csv') as f:
            info = csv.DictReader(f)
            for row in info:
                cur = []
                for attr_name in attr:
                    cur.append(0 if row[attr_name] == '-1' else 1)
                self.attr.append(np.array(cur))
                self.name.append(row['name'])
                self.orit.append(row['orientation'])
                if len(self.name) == self.num:
                    break
        
    def __len__(self):
        return self.num

    def getImage(self, index):
        img = Image.open(self.path + '/' + self.name[index])

        W, H = img.size
        bnd = (H - W) // 2
        img = img.crop((0, bnd, W, W))

        'TODO: face orientation'
        if self.orit[index] == 'right':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize((self.picSize, self.picSize), Image.ANTIALIAS)
        return (np.array(img) / 255.0).transpose(2, 0, 1)

    def __getitem__(self, item):
        """
        :param item:index of the pic
        :return: (picture(numpy 3 * sz * sz), feat(numpy num))
        """
        return self.getImage(item), self.attr[item]


def getTestImages(path, size=128, cut=False):
    files = sorted(glob.glob((path + '/*.*')))
    print(path, files)
    ret = []
    for file in files:
        img = Image.open(file)

        if cut:
            W, H = img.size
            bbb = W // 4
            img = img.crop((bbb * 2 // 3, bbb // 4, W - bbb, W - bbb))

        img = img.resize((size, size), Image.ANTIALIAS)
        ret += [(np.array(img) / 255.0).transpose(2, 0, 1)]
    return ret

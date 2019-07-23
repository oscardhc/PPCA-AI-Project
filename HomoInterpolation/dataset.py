
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
        
        with open(path + '/celeba-with-orientation.csv') as f:
            info = csv.DictReader(f)
            for row in info:
                cur = []
                for attr_name in attr:
                    cur.append(0 if row[attr_name] == '-1' else 1)
                self.attr.append(np.array(cur))
                self.name.append(row['name'])
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

        img = img.resize((self.picSize, self.picSize))
        return (np.array(img) / 255.0).transpose(2, 0, 1)

    def __getitem__(self, item):
        """
        :param item:index of the pic
        :return: (picture(numpy 3 * sz * sz), feat(numpy num))
        """
        return self.getImage(item), self.attr[item]
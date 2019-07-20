
import glob
import random
import os

import torch
from PIL import Image
import numpy as np
import csv
import torch.utils.data

class CelebADataset(torch.utils.data.Dataset):
    
    def __init__(self, num, path, picSize, feat):
        super(CelebADataset, self).__init__()
        self.path = path
        self.num = num
        self.picSize = picSize
        self.atrs
        self.name = []
        self.feat = []
        
        with open(path + '/celeba-with-orientation.csv') as f:
            info = csv.DictReader(f)
            for row in info:
                cur = []
                for f in feat:
                    cur.append(row[f])
                self.feat.append(cur)
                self.name.append(row['name'])
        
    def __len__(self):
        return self.num
    
    def getImage(self, index):
        img = Image.open(self.faceList)
        img.resize((picSize, picsize))
        return (img.numpy() / 255.0).transpose(2, 0, 1)
    
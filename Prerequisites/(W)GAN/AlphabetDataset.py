import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from random import random


class Alphabet(Dataset):

    def __init__(self, num):
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        path = "/home/oscar/dhc/Danbooru/AnimeHeadDetector/det/i%d.jpg" % (idx + 1)
        img = Image.open(path).resize((128, 128), Image.ANTIALIAS)
        img.convert('RGB')
        arr = np.array(img) / 255.0
        return (arr.transpose(2, 0, 1), random() * 0.3 + 0.7)

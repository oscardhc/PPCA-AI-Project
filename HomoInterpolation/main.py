
import torch
from dataset import CelebADataset
import train

if __name__ == '__main__':
    a = CelebADataset(100, '../../celeba-dataset', 128)
    
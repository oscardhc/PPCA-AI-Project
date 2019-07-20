
import torch
from dataset import CelebADataset

if __name__ == '__main__':
    a = CelebADataset(100, '../../celeba-dataset', 128)
    
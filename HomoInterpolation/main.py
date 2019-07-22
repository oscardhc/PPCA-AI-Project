
import torch
from dataset import CelebADataset
import train

if __name__ == '__main__':
    a = CelebADataset(100, '../../celeba-dataset', 128, ['Smiling', 'Young', 'Blond_Hair'])
    it = train.Program(device=('cuda:0' if torch.cuda.is_available() else 'cpu'))
    it.train(device=('cuda:0' if torch.cuda.is_available() else 'cpu'))
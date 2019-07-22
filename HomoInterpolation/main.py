
import torch
from dataset import CelebADataset
import train

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    it = train.Program(device=device, attr=['Mouth_Slightly_Open', 'Young', 'Blond_Hair', 'Male'])
    it.train()
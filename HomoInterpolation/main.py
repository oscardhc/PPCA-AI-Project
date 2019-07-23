
import torch
import train

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    onServer = False
    it = train.Program(imgsize=128, toLoad=False, device=device,
                       attr=['Mouth_Slightly_Open', 'Young', 'Blond_Hair', 'Male'], onServer=onServer)
    it.train()
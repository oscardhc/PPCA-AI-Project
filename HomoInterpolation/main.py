
import torch
import train

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    onServer = True
    attr = ['Mouth_Slightly_Open', 'Smiling',
            'Male', 'No_Beard',
            'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
            'Bald', 'Receding_Hairline', 
            'Young']
    attrGroup = [(0, 11)]
    it = train.Program(imgsize=128, toLoad=True, device=device,
                       attr=attr, onServer=onServer, attrGroup=attrGroup)
    it.train()
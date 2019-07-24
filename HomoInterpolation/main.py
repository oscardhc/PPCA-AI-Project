
import torch
import train

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    onServer = False
    attr = ['Mouth_Slightly_Open', 'Smiling',
            'Male', 'No_Beard', 'Mustache', 'Goatee', 'Sideburns',
            'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
            'Bald', 'Receding_Hairline', 'Bangs',
            'Young']
    attrGroup = [(0, 2), (2, 7), (7, 11), (11, 14), (14, 15), (0, 15)]
    it = train.Program(imgsize=128, toload=True, device=device,
                       attr=attr, onServer=onServer, attrGroup=attrGroup)
    it.train()
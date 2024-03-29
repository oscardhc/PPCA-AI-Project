
import torch
from train import Program

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    onServer = True
    print(device)
    attr = [['Mouth_Slightly_Open', 'Smiling'],
            ['Male', 'No_Beard', 'Mustache', 'Goatee', 'Sideburns'],
            ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
            ['Bald', 'Receding_Hairline', 'Bangs'],
            ['Young'],
            ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Eyeglasses'],
            ['Big_Lips', 'Big_Nose', 'Chubby', 'Double_Chin', 'High_Cheekbones', 'Narrow_Eyes', 'Pointy_Nose'],
            ['Straight_Hair', 'Wavy_Hair'],
            ['Attractive', 'Pale_Skin', 'Heavy_Makeup']]

    it = Program(imgsize=128, toLoad=True, device=device,
                 attr=attr, onServer=onServer)
#     it.train()
    it.showResult()

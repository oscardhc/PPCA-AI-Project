
import os
import json
import run
import torch

"""
bash下设置环境变量的方法（用于本地测试 最好使用绝对路径）

export TEMP_PATH="temp/"
export LONGTERM_PATH="longterm/"
export READONLY_PATH="readonly/"
"""

# 读取环境变量
TEMP_PATH = os.environ.get("TEMP_PATH")
LONGTERM_PATH = os.environ.get("LONGTERM_PATH")
READONLY_PATH = os.environ.get("READONLY_PATH")
print(TEMP_PATH)

# 配置和输出文件路径
config_path = os.path.join(TEMP_PATH, "train", "config.json")

# 读取配置文件
with open(config_path, 'r') as f:
    config = json.load(f)
print(config)

# 模型训练（略）
imagesize = int(config['image_size'])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
attr = [['Mouth_Slightly_Open', 'Smiling'],
        ['Male', 'No_Beard', 'Mustache', 'Goatee', 'Sideburns'],
        ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
        ['Bald', 'Receding_Hairline', 'Bangs'],
        ['Young'],
        ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Eyeglasses'],
        ['Big_Lips', 'Big_Nose', 'Chubby', 'Double_Chin', 'High_Cheekbones', 'Narrow_Eyes', 'Pointy_Nose'],
        ['Straight_Hair', 'Wavy_Hair'],
        ['Attractive', 'Pale_Skin', 'Heavy_Makeup']]

dataPath = READONLY_PATH + config['data_path']
testPath = READONLY_PATH + config['test_path']
vggPath = READONLY_PATH + config['vgg_path']

modelpath = LONGTERM_PATH + config['model_path']

try:
    f = open(os.path.join(LONGTERM_PATH, 'mata.txt'), 'r')
    ar = f.read().split(' ')
    f.close()
    ste = int(ar[0])
    toLoad = True
except:
    f = open(os.path.join(LONGTERM_PATH, 'mata.txt'), 'w')
    ste = 0
    print(ste, file=f)
    toLoad = False
    f.close()

epo = config['train_epoch']
batchsize = config['batch_size']
batchpen = config['batch_penalty']

prog = run.Program(imagesize, device, attr, toLoad, batchsize, epo, ste, batchpen, dataPath, testPath, modelpath, vggPath, TEMP_PATH)
prog.train()

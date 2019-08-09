import os
import json
import torch
import run
import numpy as np
from PIL import Image

TEMP_PATH = os.environ.get("TEMP_PATH")
LONGTERM_PATH = os.environ.get("LONGTERM_PATH")
READONLY_PATH = os.environ.get("READONLY_PATH")

config_path = os.path.join(TEMP_PATH, "predict", "config.json")
input_path = os.path.join(TEMP_PATH, "predict", "input.json")
output_path = os.path.join(TEMP_PATH, "predict", "output.json")

def init():
# 读取配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(config)

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

    dataPath = ''
    testPath = ''
    vggPath = ''
    
    modelpath = LONGTERM_PATH + config['model_path']

    ste = 0
    epo = 0
    batchsize = 0
    batchpen = 0

    prog = run.Program(imagesize, device, attr, True, batchsize, epo, ste, batchpen, dataPath, testPath, modelpath, vggPath, TEMP_PATH)
    return prog


def predict(prog):
    
    with open(input_path, 'r') as f:
        input_json = json.load(f)
        picA_path = os.path.join(TEMP_PATH, "predict", input_json["Apath"])
        picB_path = os.path.join(TEMP_PATH, "predict", input_json["Bpath"])
        strength = input_json["strength"]

    img = prog.predict(picA_path, picB_path, strength)
    img.save(os.path.join(TEMP_PATH, "predict", 'output.jpg'))
    
    output_json = {
        "output_path" : os.path.join(TEMP_PATH, "predict", 'output.jpg')
    }
    
    with open(output_path, "w") as f:
        json.dump(output_json, f)
        f.flush()
        
if __name__ == '__main__':
    p = init()
    predict(p)

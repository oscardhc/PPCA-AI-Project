import Lib.os as os
import Lib.json as json
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

device = 0
E = 0
D = 0
I = 0

def init():
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    enc_path = config_json["enc_path"]
    enc_path = os.path.join(LONGTERM_PATH, enc_path)
    interp_path = config_json["interp_path"]
    interp_path = os.path.join(LONGTERM_PATH, interp_path)
    dec_path = config_json["dec_path"]
    dec_path = os.path.join(LONGTERM_PATH, dec_path)

    global device, E, D, I

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
    attr_n = len(attr)

    E = m.Encoder(path=enc_path).to(device)
    D = m.Decoder().to(device)
    I = m.Interp(attr_n + 1).to(device)

    E.load_state_dict(torch.load(enc_path))
    D.load_State_dict(torch.load(dec_path))
    I.load_state_dict(torch.load(interp_path))

def predict():
    with open(input_path, 'r') as f:
        input_json = json.load(f)

    picA_path = os.path.join(TEMP_PATH, "predict", input_json["Apath"])
    picB_path = os.path.join(TEMP_PATH, "predict", input_json["Apath"])
    strenth = input_json["strenth"]
    global device, E, D, I

    A = Image.open(picA_path)
    B = Image.open(picB_path)
    A = np.array(A)
    B = np.array(B)
    a_feat = E(A)
    b_feat = E(B)
    interp_feat = I(A, B, strenth)
    out = D(interp_feat)
    out = Image.fromarray(out)
    picOut_path = os.path.join(TEMP_PATH, "predict", "output.jpg")
    out.save(picOut_path)

    output_json = {
        "output_path" : "output.jpg"
    }
    with open(output_path, "w") as f:
        json.dump(output_json, f)
        f.flush()


if __name__ == "__main__":
    init()
    predict()
import numpy as np 
from PIL import Image
import numpy as np
import Lib.json as json
import Lib.os as os

# assume a and b are two numpys representing two pictures, 
# they are all in size of 128*128*3
# the result is saved in another numpy named res


TEMP_PATH = os.environ.get("TEMP_PATH")
LONGTERM_PATH = os.environ.get("LONGTERM_PATH")
READONLY_PATH = os.environ.get("READONLY_PATH")


config_path = os.path.join(TEMP_PATH, "predict", "config.json")
input_path = os.path.join(TEMP_PATH, "predict", "input.json")
output_path = os.path.join(TEMP_PATH, "predict", "output.json")

def init():
    # since this algorithm does not have nn, nothing needs to be initialized
    pass

def predict():
    a = Image.open('inputA.jpg')
    b = Image.open('inputB.jpg')
    strenth = input["strenth"]
    a = np.array(a)
    b = np.array(b)
    res = np.zeros_like(a)

    h = a.shape[0]
    w = a.shape[1]

    for i in range(h):
        for j in range(w):
            for k in range(3):
                res[i][j][k] = a[i][j][k] * (1 - strenth) + b[i][j][k] * strenth

    res = Image.fromarray(res.astype('uint8')).convert('RGB')
    picOut_path = os.path.join(TEMP_PATH, "predict", "output.jpg")
    res.save(picOut_path)
    output_json = {
        "output_path" : "output.jpg"
    }
    with open(output_path, "w") as f:
        json.dump(output_json, f)
        f.flush()

if __name__ == "__main__":
    init()
    predict()
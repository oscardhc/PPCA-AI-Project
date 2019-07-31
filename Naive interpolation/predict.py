import numpy as np 
from PIL import Image
import numpy as np
import Lib.json as json

# assume a and b are two numpys representing two pictures, 
# they are all in size of 128*128*3
# the result is saved in another numpy named res


# TEMP_PATH = os.environ.get("TEMP_PATH")
# LONGTERM_PATH = os.environ.get("LONGTERM_PATH")
# READONLY_PATH = os.environ.get("READONLY_PATH")


# config_path = os.path.join(TEMP_PATH, "predict", "config.json")
# input_path = os.path.join(TEMP_PATH, "predict", "input.json")
# output_path = os.path.join(TEMP_PATH, "predict", "output.json")

a = Image.open('inputA.jpg')
b = Image.open('inputB.jpg')
strenth = 0# input["strenth"]
a = np.array(a)
b = np.array(b)
res = np.zeros_like(a)

w = 128
h = 128

for _ in range(4):
    strenth += 0.2
    for i in range(w):
        for j in range(h):
            for k in range(3):
                res[i][j][k] = a[i][j][k] * strenth + b[i][j][k] * (1 - strenth)

res = Image.fromarray(res.astype('uint8')).convert('RGB')
res.save('output.jpg')
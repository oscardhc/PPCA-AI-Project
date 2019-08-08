
import os
import json
import run

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

# 配置和输出文件路径
config_path = os.path.join(TEMP_PATH, "train", "config.json")
output_path = os.path.join(TEMP_PATH, "train", "output.json")

# 读取配置文件
with open(config_path, 'r') as f:
    config = json.load(f)
print(config)

# 数据集路径
dataset_path = os.path.join(READONLY_PATH, config['dataset_path'])

# 模型训练（略）


# 保存模型和结果
model_name = "model.h5"
model_path = os.path.join(LONGTERM_PATH, model_name)

# model.save(model_path)
output_json = {
    "model_path" : model_name, # 相对于LONGTERM_PATH的路径
    "iterations" : "", # 训练中间结果
    "val_err"  : 0.15
}

with open(output_path, "w") as f:
    json.dump(output_json, f)
    f.flush() # 中间结果更新时需要flush
import pandas as pd
import json

# 文件路径
json_file_path = "/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/grpo_1B_xiangxiguize/only_kg_to_answer/tqa_answer.json"
n = 0
x = 0

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data_list = json.load(json_file)
    
    for data in data_list:
        n += 1
        ground_truth = data.get('ground_truth', [])
        model_answer = data.get('model_answer', '')
        
        # 如果ground_truth是列表，检查其中是否有任何一个答案出现在model_answer中
        if isinstance(ground_truth, list):
            if any(str(truth).lower() in str(model_answer).lower() for truth in ground_truth if truth):
                x += 1
        # 如果ground_truth是字符串
        elif isinstance(ground_truth, str) and ground_truth:
            if ground_truth.lower() in str(model_answer).lower():
                x += 1

print(f"匹配数量: {x}")
print(f"总数量: {n}")
print(f"匹配比例: {x/n:.4f} ({x/n*100:.2f}%)")
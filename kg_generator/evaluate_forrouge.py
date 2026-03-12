import json
from rouge import Rouge

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
        return scores["rouge-l"]["f"]
    except ValueError:  # Hypothesis is empty
        return 0.0

def calculate_average_rougel(json_file):
    total_score = 0.0
    num_questions = 0

    # 读取JSON文件
    with open(json_file, "r", encoding="utf-8") as file:
        data_list = json.load(file)
    
    # 遍历JSON列表中的每个数据项
    for data in data_list:
        model_answer = data.get("model_answer", "")
        ground_truths = data.get("ground_truth", [])
        
        # 如果ground_truths是字符串，转换为列表
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        
        # 计算与每个ground_truth的ROUGE-L分数，取最大值
        if ground_truths:
            max_score = max(_rougel_score(model_answer, gt) for gt in ground_truths)
        else:
            max_score = 0.0
        
        total_score += max_score
        num_questions += 1

    return total_score / num_questions if num_questions > 0 else 0.0

# 使用示例
json_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/grpo_1B_xiangxiguize/only_kg_to_answer/marco_answer.json"  
average_rougel = calculate_average_rougel(json_file)
print(f"Average ROUGE-L score: {average_rougel:.4f}")
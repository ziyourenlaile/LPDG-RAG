import json
import re
from rouge import Rouge

def _rougel_score(prediction, ground_truth):
    """计算ROUGE-L分数"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:
        return 0.0
    return scores["rouge-l"]["f"]

def is_answer_correct(data_type, ground_truth, model_answer):
    """
    判断答案是否正确
    
    Args:
        data_type: 数据类型
        ground_truth: 标准答案
        model_answer: 模型答案
    
    Returns:
        bool: 答案是否正确
    """
    # 第一类数据类型：直接检查ground_truth是否在model_answer中
    if data_type in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa', 'gsm8k', 'strategyqa', 'web_questions']:
        match_result = ground_truth in model_answer
        return match_result
    # 第二类数据类型：ROUGE-L评分
    elif data_type in ['wiki_qa', 'yahoo_answers_qa', 'marcoqa']:
        score = _rougel_score(model_answer, ground_truth)
        return score > 0.22
    # 其他数据类型默认返回False
    else:
        return False

def filter_correct_answers(input_file, output_file):
    """
    筛选包含正确答案的数据
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    correct_data = []
    
    for item in data:
        data_type = item.get("data_type", "").lower()
        ground_truth = item.get("answer", "")
        model_answer = item.get("model_answer", "")
        
        # 如果没有model_answer或ground_truth，跳过
        if not model_answer or not ground_truth:
            continue
        
        # 使用简化的判断逻辑
        if is_answer_correct(data_type, ground_truth, model_answer):
            correct_data.append(item)
    
    # 保存筛选后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(correct_data, f, ensure_ascii=False, indent=2)
    
    print(f"原始数据数量: {len(data)}")
    print(f"筛选后正确数据数量: {len(correct_data)}")
    print(f"正确数据已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    input_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_yuan/kg_to_ans.json"
    output_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_yuan/correct_kg_to_ans.json"
    
    filter_correct_answers(input_file, output_file)
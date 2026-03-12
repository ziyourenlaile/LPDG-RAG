import json
from rouge import Rouge

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def is_answer_correct(data_type, model_answer, ground_truth):
    """
    判断模型答案是否正确
    """
    # 处理可能的None值
    if model_answer is None or ground_truth is None:
        return False
        
    model_answer = str(model_answer).lower().strip()
    ground_truth = str(ground_truth).lower().strip()  
    
    # 第一类数据类型：精确匹配
    if data_type in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa', 'gsm8k', 'strategyqa', 'web_questions']:
        match_result = ground_truth in model_answer
        return match_result
    # 第二类数据类型：ROUGE-L评分
    elif data_type in ['wiki_qa', 'yahoo_answers_qa', 'marcoqa']:
        score = _rougel_score(model_answer, ground_truth)
        return score > 0.30
    else:
        # 未知数据类型，默认使用精确匹配
        match_result = ground_truth in model_answer
        return match_result

def custom_json_decoder(dct):
    """
    自定义JSON解码器
    """
    # 这里可以添加自定义的解码逻辑
    # 例如：处理特殊字段、数据类型转换等
    return dct

def process_jsonl_file(input_file, output_file):
    """
    处理JSONL文件，为每个条目添加is_correct字段
    """
    processed_count = 0
    correct_count = 0
    
    # 加载数据并合并重复问题
    with open(input_file, 'r') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
    
    # 处理每条数据
    processed_data = []
    for item in data:
        # 判断答案是否正确
        is_correct = is_answer_correct(
            item.get('data_type'),
            item.get('model_answer'),
            item.get('ground_truth')
        )
        
        # 添加is_correct字段
        item['is_correct'] = is_correct
        
        # 统计正确数量
        if is_correct:
            correct_count += 1
        
        processed_data.append(item)
        processed_count += 1
    
    # 写入输出文件
    with open(output_file, 'w') as outfile:
        for item in processed_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"处理完成！共处理 {processed_count} 条数据，其中 {correct_count} 条正确，正确率: {correct_count/processed_count*100:.2f}%")


# 测试您提供的示例数据
if __name__ == "__main__":    
    # 处理整个文件（请修改为实际的文件路径）
    input_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/data/queryCoT_to_answer_12_7.jsonl"
    output_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/data/queryCoT_to_answer_12_7_with_correct.jsonl"
    
    # 执行文件处理
    process_jsonl_file(input_file, output_file)
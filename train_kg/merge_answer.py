# import json
# import os
# import re
# from collections import defaultdict
# from rouge_score import rouge_scorer

# def load_json_files(file_paths):
#     """加载三个JSON文件"""
#     data_dict = {}
    
#     for file_path in file_paths:
#         level = file_path.split('_')[-1].replace('.json', '')  # 提取级别：few, medium, many
#         print(f"正在加载 {level} 文件: {file_path}")
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             data_dict[level] = data
#             print(f"  {level} 文件加载成功，共 {len(data)} 条数据")
#         except Exception as e:
#             print(f"  {level} 文件加载失败: {e}")
#             return None
    
#     return data_dict

# def _rouge1_score(answer, model_answer):
#     """计算ROUGE-1 F1分数"""
#     if not answer or not model_answer:
#         return 0.0
    
#     scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
#     scores = scorer.score(answer, model_answer)
#     return scores['rouge1'].fmeasure

# def _exact_match_score(answer, model_answer):
#     """精确匹配检查，避免部分匹配问题"""
#     if not answer or not model_answer:
#         return 0.0
    
#     # 清理答案文本
#     def clean_text(text):
#         # 移除标点符号和多余空格
#         text = re.sub(r'[^\w\s]', '', text.lower().strip())
#         # 合并多个空格
#         text = re.sub(r'\s+', ' ', text)
#         return text
    
#     clean_answer = clean_text(answer)
#     clean_model_answer = clean_text(model_answer)
    
#     # 检查是否完全匹配
#     if clean_answer == clean_model_answer:
#         return 1.0
    
#     # 检查答案是否作为完整单词出现在模型答案中
#     answer_words = clean_answer.split()
#     model_answer_words = clean_model_answer.split()
    
#     # 如果答案是单个单词，检查是否作为完整单词出现在模型答案中
#     if len(answer_words) == 1:
#         return 1.0 if answer_words[0] in model_answer_words else 0.0
    
#     # 如果答案是多个单词，检查是否作为连续短语出现在模型答案中
#     answer_phrase = ' '.join(answer_words)
#     model_answer_text = ' '.join(model_answer_words)
#     return 1.0 if answer_phrase in model_answer_text else 0.0

# def _math_answer_match(answer, model_answer):
#     """数学答案匹配，支持多种格式"""
#     if not answer or not model_answer:
#         return 0.0
    
#     # 清理数学答案
#     def clean_math_answer(text):
#         # 移除括号、空格等
#         text = re.sub(r'[\(\)\[\]\s]', '', text.lower().strip())
#         # 提取数字和运算符
#         return text
    
#     clean_answer = clean_math_answer(answer)
#     clean_model_answer = clean_math_answer(model_answer)
    
#     # 直接匹配
#     if clean_answer == clean_model_answer:
#         return 1.0
    
#     # 检查数字匹配
#     answer_numbers = re.findall(r'\d+\.?\d*', answer)
#     model_numbers = re.findall(r'\d+\.?\d*', model_answer)
    
#     if answer_numbers and model_numbers:
#         # 检查是否有共同的数字
#         common_numbers = set(answer_numbers) & set(model_numbers)
#         if common_numbers:
#             return 0.5  # 部分匹配
    
#     return 0.0

# def _contains_match(answer, model_answer):
#     """简单的包含匹配"""
#     if not answer or not model_answer:
#         return False
    
#     # 将答案和模型答案都转换为小写进行比较
#     answer_lower = answer.lower().strip()
#     model_answer_lower = model_answer.lower()
    
#     return answer_lower in model_answer_lower

# def _count_words(text):
#     """计算文本中的单词数量"""
#     if not text:
#         return 0
#     words = re.findall(r'\b\w+\b', text)
#     return len(words)

# def _contains_numbers(text):
#     """检查文本中是否包含数字"""
#     if not text:
#         return False
#     return bool(re.search(r'\d', text))

# def check_answer_in_model_answer(answer, model_answer):
#     """检查答案是否在模型生成的答案中（智能版本）"""
#     if not answer or not model_answer:
#         return False
    
#     # 计算答案的单词数量
#     answer_word_count = _count_words(answer)
    
#     # 如果答案单词数大于5，使用ROUGE-1评分
#     if answer_word_count > 5:
#         rouge_score = _rouge1_score(answer, model_answer)
#         return rouge_score >= 0.22  # ROUGE分数阈值可以调整
    
#     # 对于短答案（<=5个单词）
#     else:
#         # 对于不包含数字的短答案，使用精确匹配
#         exact_score = _exact_match_score(answer, model_answer)
#         if exact_score >= 0.5:
#             return True
        
#         # 最后使用简单的包含匹配作为后备
#         return False

# def process_three_levels(data_dict):
#     """处理三个级别的数据"""
#     # 按 question_id + passage_id 分组
#     question_passage_groups = defaultdict(dict)
    
#     # 将数据按 question_id + passage_id 分组
#     for level, data in data_dict.items():
#         for item in data:
#             # 使用 question_id 和 passage_id 的组合作为唯一标识
#             unique_id = f"{item['question_id']}_{item['passage_id']}"
#             question_passage_groups[unique_id][level] = item
    
#     # 处理每个唯一标识的数据
#     selected_data = []
    
#     for unique_id, levels_data in question_passage_groups.items():
#         # 检查三个级别是否都存在
#         if 'few' not in levels_data or 'medium' not in levels_data or 'many' not in levels_data:
#             continue
        
#         few_data = levels_data['few']
#         medium_data = levels_data['medium']
#         many_data = levels_data['many']
        
#         # 检查哪些级别的答案在对应的model_answer中
#         few_match = check_answer_in_model_answer(few_data['answer'], few_data['model_answer'])
#         medium_match = check_answer_in_model_answer(medium_data['answer'], medium_data['model_answer'])
#         many_match = check_answer_in_model_answer(many_data['answer'], many_data['model_answer'])
        
#         # 收集正确的级别
#         correct_levels = []
#         if few_match:
#             correct_levels.append(('few', few_data))
#         if medium_match:
#             correct_levels.append(('medium', medium_data))
#         if many_match:
#             correct_levels.append(('many', many_data))
        
#         # 如果有至少一个级别正确
#         if correct_levels:
#             # 按照triple_count排序，选择最小的
#             correct_levels.sort(key=lambda x: x[1]['triple_count'])
#             selected_data.append(correct_levels[0][1])
    
#     return selected_data

# def save_selected_data(selected_data, output_path):
#     """保存筛选后的数据"""
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(selected_data, f, ensure_ascii=False, indent=2)
    
#     print(f"筛选后的数据已保存到: {output_path}")

# def analyze_selection(data_dict, selected_data):
#     """分析选择情况"""
#     correct_count_by_level = {'few': 0, 'medium': 0, 'many': 0}
#     total_selected = len(selected_data)
    
#     for item in selected_data:
#         level = item['level']
#         correct_count_by_level[level] += 1
    
#     print(f"\n=== 选择分析 ===")
#     print(f"总选择数据数量: {total_selected}")
#     for level in ['few', 'medium', 'many']:
#         count = correct_count_by_level[level]
#         percentage = count / total_selected * 100 if total_selected > 0 else 0
#         print(f"{level} 级别选择数量: {count} ({percentage:.1f}%)")

# def main():
#     # 文件路径
#     file_paths = [
#         '/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_to_ans_few.json',
#         '/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_to_ans_medium.json',
#         '/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_to_ans_many.json'
#     ]
    
#     # 输出路径
#     output_path = '/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_selected_answer.json'
    
#     # 加载文件
#     print("开始加载三个级别的数据文件...")
#     data_dict = load_json_files(file_paths)
    
#     if not data_dict:
#         print("文件加载失败，程序退出")
#         return
    
#     # 处理数据
#     print("\n开始处理数据...")
#     selected_data = process_three_levels(data_dict)
    
#     # 保存结果
#     print(f"\n=== 处理结果 ===")
    
#     # 计算唯一标识的数量（question_id + passage_id）
#     unique_ids_by_level = {}
#     for level, data in data_dict.items():
#         unique_ids = set(f"{item['question_id']}_{item['passage_id']}" for item in data)
#         unique_ids_by_level[level] = unique_ids
    
#     total_unique_ids = set().union(*unique_ids_by_level.values())
#     print(f"总唯一标识数量 (question_id+passage_id): {len(total_unique_ids)}")
    
#     # 计算三个级别都存在的唯一标识数量
#     common_unique_ids = set.intersection(*unique_ids_by_level.values())
#     print(f"三个级别都存在的唯一标识数量: {len(common_unique_ids)}")
#     print(f"筛选后的数据数量: {len(selected_data)}")
    
#     # 统计信息
#     if selected_data:
#         triple_counts = [item['triple_count'] for item in selected_data]
#         print(f"三元组数量范围: {min(triple_counts)}-{max(triple_counts)}")
#         print(f"平均三元组数量: {sum(triple_counts)/len(triple_counts):.2f}")
        
#         # 分析选择情况
#         analyze_selection(data_dict, selected_data)
    
#     save_selected_data(selected_data, output_path)

# if __name__ == "__main__":
#     main()

import json
import os
import re
from collections import defaultdict
from rouge import Rouge

def load_json_files(file_paths):
    """加载三个JSON文件"""
    data_dict = {}
    
    for file_path in file_paths:
        level = file_path.split('_')[-1].replace('.json', '')  # 提取级别：few, medium, many
        print(f"正在加载 {level} 文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data_dict[level] = data
            print(f"  {level} 文件加载成功，共 {len(data)} 条数据")
        except Exception as e:
            print(f"  {level} 文件加载失败: {e}")
            return None
    
    return data_dict

def _rougel_score(prediction, ground_truth):
    """计算ROUGE-L分数"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:
        return 0.0
    return scores["rouge-l"]["f"]

def check_answer_correct(ground_truth, model_answer):
    """
    检查答案是否正确
    
    Args:
        ground_truth: 标准答案
        model_answer: 模型答案
    
    Returns:
        bool: 答案是否正确
    """
    if not ground_truth or not model_answer:
        return False
    
    # 计算标准答案的单词数量
    def count_words(text):
        if not text:
            return 0
        # 简单的单词计数，按空格分割
        return len(text.strip().split())
    
    gt_word_count = count_words(ground_truth)
    
    # 根据单词数量判断类型
    if gt_word_count <= 10:
        # 短答案（第一类）：使用精确匹配
        match_result = ground_truth.strip().lower() in model_answer.lower()
        return match_result
    else:
        # 长答案（第二类）：使用ROUGE-L评分
        score = _rougel_score(model_answer, ground_truth)
        return score > 0.22

def process_three_levels(data_dict):
    """处理三个级别的数据"""
    # 按 question_id + passage_id 分组
    question_passage_groups = defaultdict(dict)
    
    # 将数据按 question_id + passage_id 分组
    for level, data in data_dict.items():
        for item in data:
            # 使用 question_id 和 passage_id 的组合作为唯一标识
            unique_id = f"{item['question_id']}_{item['passage_id']}"
            question_passage_groups[unique_id][level] = item
    
    # 处理每个唯一标识的数据
    selected_data = []
    
    for unique_id, levels_data in question_passage_groups.items():
        # 检查三个级别是否都存在
        if 'few' not in levels_data or 'medium' not in levels_data or 'many' not in levels_data:
            continue
        
        few_data = levels_data['few']
        medium_data = levels_data['medium']
        many_data = levels_data['many']
        
        # 获取数据类型
        # data_type = few_data.get('data_type', '').lower()
        
        # 检查哪些级别的答案在对应的model_answer中
        few_match = check_answer_correct(few_data['answer'], few_data['model_answer'])
        medium_match = check_answer_correct(medium_data['answer'], medium_data['model_answer'])
        many_match = check_answer_correct(many_data['answer'], many_data['model_answer'])
        
        # 收集正确的级别
        correct_levels = []
        if few_match:
            correct_levels.append(('few', few_data))
        if medium_match:
            correct_levels.append(('medium', medium_data))
        if many_match:
            correct_levels.append(('many', many_data))
        
        # 如果有至少一个级别正确
        if correct_levels:
            # 按照triple_count排序，选择最小的
            correct_levels.sort(key=lambda x: x[1]['triple_count'])
            selected_data.append(correct_levels[0][1])
    
    return selected_data

def save_selected_data(selected_data, output_path):
    """保存筛选后的数据"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)
    
    print(f"筛选后的数据已保存到: {output_path}")

def analyze_selection(data_dict, selected_data):
    """分析选择情况"""
    correct_count_by_level = {'few': 0, 'medium': 0, 'many': 0}
    total_selected = len(selected_data)
    
    for item in selected_data:
        level = item['level']
        correct_count_by_level[level] += 1
    
    print(f"\n=== 选择分析 ===")
    print(f"总选择数据数量: {total_selected}")
    for level in ['few', 'medium', 'many']:
        count = correct_count_by_level[level]
        percentage = count / total_selected * 100 if total_selected > 0 else 0
        print(f"{level} 级别选择数量: {count} ({percentage:.1f}%)")

def main():
    # 文件路径
    file_paths = [
        '/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_to_ans_few.json',
        '/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_to_ans_medium.json',
        '/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_to_ans_many.json'
    ]
    
    # 输出路径
    output_path = '/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_selected_answer_new.json'
    
    # 加载文件
    print("开始加载三个级别的数据文件...")
    data_dict = load_json_files(file_paths)
    
    if not data_dict:
        print("文件加载失败，程序退出")
        return
    
    # 处理数据
    print("\n开始处理数据...")
    selected_data = process_three_levels(data_dict)
    
    # 保存结果
    print(f"\n=== 处理结果 ===")
    
    # 计算唯一标识的数量（question_id + passage_id）
    unique_ids_by_level = {}
    for level, data in data_dict.items():
        unique_ids = set(f"{item['question_id']}_{item['passage_id']}" for item in data)
        unique_ids_by_level[level] = unique_ids
    
    total_unique_ids = set().union(*unique_ids_by_level.values())
    print(f"总唯一标识数量 (question_id+passage_id): {len(total_unique_ids)}")
    
    # 计算三个级别都存在的唯一标识数量
    common_unique_ids = set.intersection(*unique_ids_by_level.values())
    print(f"三个级别都存在的唯一标识数量: {len(common_unique_ids)}")
    print(f"筛选后的数据数量: {len(selected_data)}")
    
    # 统计信息
    if selected_data:
        triple_counts = [item['triple_count'] for item in selected_data]
        print(f"三元组数量范围: {min(triple_counts)}-{max(triple_counts)}")
        print(f"平均三元组数量: {sum(triple_counts)/len(triple_counts):.2f}")
        
        # 分析选择情况
        analyze_selection(data_dict, selected_data)
    
    save_selected_data(selected_data, output_path)

if __name__ == "__main__":
    main()

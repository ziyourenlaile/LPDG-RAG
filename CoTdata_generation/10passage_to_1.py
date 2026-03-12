import json
import re
from rouge import Rouge
from collections import defaultdict

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
    
    # 未知数据类型
    else:
        # print(f"Warning: Unknown data_type '{data_type}', using exact match")
        pattern = r'\b' + re.escape(ground_truth) + r'\b'
        return bool(re.search(pattern, model_answer))

def process_and_merge_jsonl_file(input_filepath, output_filepath):
    """
    处理JSONL文件，合并相同ID的记录，并重新组织数据结构
    """
    # 用于合并相同ID的记录
    merged_data = defaultdict(list)
    processed_count = 0
    error_count = 0
    
    print("第一步：读取和解析数据...")
    
    # 第一步：读取所有数据并按ID分组
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析单行JSON对象
                item = json.loads(line)
                
                if not isinstance(item, dict):
                    error_count += 1
                    continue
                
                item_id = item.get('id', '')
                if not item_id:
                    print(f"Warning: Line {line_num} has no ID, skipping")
                    error_count += 1
                    continue
                
                # 添加到合并数据中
                merged_data[item_id].append(item)
                processed_count += 1
                
                # # 进度显示
                # if line_num % 10000 == 0:
                #     print(f"已读取 {line_num} 行，成功解析: {processed_count}，错误: {error_count}")
                #     break
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
                continue
    
    print(f"\n数据读取完成！")
    print(f"成功解析: {processed_count} 条记录")
    print(f"错误行数: {error_count}")
    print(f"唯一ID数量: {len(merged_data)}")
    
    print("\n第二步：合并数据和计算正确性...")
    
    # 第二步：处理合并后的数据
    final_results = []
    correct_total_count = 0
    
    for item_id, items in merged_data.items():
        try:
            # 使用第一个item作为基础信息
            base_item = items[0]
            
            # 构建新的数据结构
            new_item = {
                "id": item_id,
                "data_type": base_item.get('data_type', ''),
                "query": base_item.get('query', ''),
                "ground_truth": base_item.get('ground_truth', ''),
                "correct_passages": [],
                "passages": []
            }
            
            # 处理每个passage
            correct_passage_ids = []
            for item in items:
                passage_info = {
                    "id": "",
                    "segment": "",
                    "COT": "",
                    "model_answer": "",
                    "is_correct": False
                }
                
                # 提取passage的id和segment
                if 'passage' in item and isinstance(item['passage'], dict):
                    passage_info["id"] = item['passage'].get('id', '')
                    passage_info["segment"] = item['passage'].get('segment', '')
                
                # 提取其他字段
                passage_info["COT"] = item.get('COT', '')
                passage_info["model_answer"] = item.get('model_answer', '')
                
                # 判断答案是否正确
                data_type = new_item['data_type']
                model_answer = passage_info["model_answer"]
                ground_truth = new_item['ground_truth']
                is_correct = is_answer_correct(data_type, model_answer, ground_truth)
                
                passage_info["is_correct"] = is_correct
                
                # 如果正确，添加到correct_passage列表
                if is_correct and passage_info["id"]:
                    correct_passage_ids.append(passage_info["id"])
                
                new_item["passages"].append(passage_info)
            
            # 设置correct_passage字段
            new_item["correct_passages"] = correct_passage_ids
            correct_total_count += len(correct_passage_ids)
            
            final_results.append(new_item)
            
        except Exception as e:
            print(f"Error merging item {item_id}: {e}")
            error_count += 1
            continue
    
    print(f"\n数据处理完成！")
    print(f"生成合并记录: {len(final_results)} 条")
    print(f"总正确答案数量: {correct_total_count}")
    
    print("\n第三步：写入输出文件...")
    
    # 第三步：写入输出文件
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for item in final_results:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"输出文件已保存: {output_filepath}")
    
    # 统计信息
    passages_per_item = [len(item['passages']) for item in final_results]
    correct_per_item = [len(item['correct_passages']) for item in final_results]
    
    print(f"\n统计信息:")
    print(f"平均每个问题的passage数量: {sum(passages_per_item)/len(passages_per_item):.2f}")
    print(f"平均每个问题的正确答案数量: {sum(correct_per_item)/len(correct_per_item):.2f}")
    print(f"有正确答案的问题数量: {sum(1 for item in final_results if item['correct_passages'])}")
    print(f"没有正确答案的问题数量: {sum(1 for item in final_results if not item['correct_passages'])}")

# 使用示例
if __name__ == "__main__":
    input_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/data/queryCoT_to_answer_12_7_with_correct.jsonl"
    output_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/data/queryCoT_to_answer_12_7_merged.jsonl"
    
    print("开始处理文件...")
    process_and_merge_jsonl_file(input_file, output_file)
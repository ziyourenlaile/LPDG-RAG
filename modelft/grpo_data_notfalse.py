import json

def filter_data(input_file, output_file):
    """筛选数据，保留model_self_correct为true或correct_passages有内容的数据"""
    
    filtered_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    
                    # 检查条件：model_self_correct为True 或 correct_passages不为空
                    if (data.get('correct_passages')):
                        filtered_data.append(data)
                        
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}, 行内容: {line[:100]}...")
    
    # 保存筛选后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"原始数据条数: 需要统计")
    print(f"筛选后数据条数: {len(filtered_data)}")
    # print(f"筛选条件: model_self_correct为True 或 correct_passages不为空")
    
    # # 统计信息
    # self_correct_count = sum(1 for data in filtered_data if (data.get('model_self_correct') is True and not data.get('correct_passages')))
    # has_passages_count = sum(1 for data in filtered_data if (data.get('model_self_correct') is False and data.get('correct_passages')))
    
    # print(f"满足(data.get('model_self_correct') is True and not data.get('correct_passages')的数据: {self_correct_count}条")
    # print(f"满足(data.get('model_self_correct') is False and data.get('correct_passages'))不为空的数据: {has_passages_count}条")

# 使用示例
if __name__ == "__main__":
    input_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/data/final_merged_with_correctness_12_7.jsonl"
    output_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/modelft/data/data_grpo/only_true.jsonl"
    
    filter_data(input_file, output_file)
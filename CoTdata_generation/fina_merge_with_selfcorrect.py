import json
from collections import defaultdict

def merge_jsonl_files(merged_filepath, correctness_filepath, output_filepath):
    """
    合并两个JSONL文件
    - merged_filepath: 包含passage信息的合并文件
    - correctness_filepath: 包含模型自身正确性的文件
    - output_filepath: 输出文件路径
    """
    
    print("第一步：加载模型自身正确性数据...")
    # 加载question_to_answer_with_correctness.jsonl文件
    model_correctness = {}
    model_correct_output = {}
    with open(correctness_filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # 作为JSON数组处理
            data = json.loads(content)
            for item in data:
                if isinstance(item, dict) and 'id' in item:
                    model_correctness[item['id']] = item.get('is_correct', False)
                    model_correct_output[item['id']] = item.get('model_answer', False)
            print(f"从JSON数组加载 {len(data)} 条记录")
        else:
            # 作为JSONL处理
            f.seek(0)
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and 'id' in item:
                        model_correctness[item['id']] = item.get('is_correct', False)
                        model_correct_output[item['id']] = item.get('model_answer', False)
                        count += 1
                except json.JSONDecodeError as e:
                    print(f"解析错误: {e}")
                    continue
            print(f"从JSONL加载 {count} 条记录")
    
    print(f"成功加载 {len(model_correctness)} 个ID的模型正确性信息")
    
    print("\n第二步：加载并合并merged文件...")
    # 加载merged文件并添加模型正确性字段
    merged_count = 0
    updated_count = 0
    
    with open(merged_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                
                if not isinstance(item, dict) or 'id' not in item:
                    continue
                
                item_id = item['id']
                
                # 添加模型自身正确性字段
                if item_id in model_correctness:
                    item['model_self_correct'] = model_correctness[item_id]
                    item['model_self_answer'] = model_correct_output[item_id]
                    updated_count += 1
                else:
                    item['model_self_correct'] = False  # 如果没有找到，默认为False
                    item['model_self_answer'] = ""
                
                # 写入更新后的数据
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                merged_count += 1
                
                # 进度显示
                if line_num % 100 == 0:
                    print(f"已处理 {line_num} 行，成功合并: {merged_count}")
                
            except json.JSONDecodeError as e:
                print(f"解析merged文件第{line_num}行错误: {e}")
                continue
            except Exception as e:
                print(f"处理merged文件第{line_num}行错误: {e}")
                continue
    
    print(f"\n合并完成！")
    print(f"成功处理记录: {merged_count} 条")
    print(f"成功添加模型正确性信息: {updated_count} 条")
    print(f"输出文件: {output_filepath}")
    
    # 统计信息
    if merged_count > 0:
        correctness_stats = {
            'model_self_correct_true': 0,
            'model_self_correct_false': 0
        }
        
        with open(output_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get('model_self_correct'):
                    correctness_stats['model_self_correct_true'] += 1
                else:
                    correctness_stats['model_self_correct_false'] += 1
        
        print(f"\n统计信息:")
        print(f"模型自身能正确回答的问题: {correctness_stats['model_self_correct_true']}")
        print(f"模型自身不能正确回答的问题: {correctness_stats['model_self_correct_false']}")
        print(f"正确率: {correctness_stats['model_self_correct_true']/merged_count*100:.2f}%")

# 使用示例
if __name__ == "__main__":
    # 文件路径
    merged_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/data/queryCoT_to_answer_12_7_merged.jsonl"
    correctness_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/data/question_to_answer_with_correctness.jsonl"
    output_file = "/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/data/final_merged_with_correctness_12_7.jsonl"
    
    print("开始合并文件...")
    merge_jsonl_files(merged_file, correctness_file, output_file)
import json
from collections import defaultdict

def merge_asqa_data(dev_file_path, query_to_answer_file_path, output_file_path):
    
    # 读取asqa_dev.jsonl数据，按sample_id组织
    dev_data = {}
    with open(dev_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            sample_id = data.get('sample_id')
            if sample_id:
                dev_data[sample_id] = data
    
    print(f"从asqa_dev.jsonl读取了 {len(dev_data)} 条数据")
    
    # 读取并更新asqa_query_to_answer.jsonl
    updated_data = []
    with open(query_to_answer_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            sample_id = data.get('id')
            
            # 如果找到对应的dev数据，添加qa_pairs信息
            if sample_id in dev_data:
                dev_item = dev_data[sample_id]
                # 添加qa_pairs到当前数据
                data['qa_pairs'] = dev_item.get('qa_pairs', [])
                # 也可以选择添加其他字段，比如wikipages, annotations等
                # data['wikipages'] = dev_item.get('wikipages', [])
                # data['annotations'] = dev_item.get('annotations', [])
            
            updated_data.append(data)
    
    print(f"处理了 {len(updated_data)} 条asqa_query_to_answer.jsonl数据")
    
    # 保存更新后的数据
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"合并后的数据已保存到: {output_file_path}")
    
    # 统计信息
    matched_count = sum(1 for item in updated_data if 'qa_pairs' in item)
    print(f"成功匹配并添加qa_pairs的数据: {matched_count} 条")

def verify_merge(output_file_path, num_samples=2):
    """
    验证合并结果
    """
    print("\n验证合并结果:")
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < num_samples:
                data = json.loads(line.strip())
                print(f"\n样本 {i+1}:")
                print(f"  ID: {data.get('id')}")
                print(f"  Query: {data.get('query')}")
                print(f"  是否包含qa_pairs: {'qa_pairs' in data}")
                if 'qa_pairs' in data:
                    print(f"  qa_pairs数量: {len(data['qa_pairs'])}")
                    for j, qa_pair in enumerate(data['qa_pairs']):
                        print(f"    QA对 {j+1}: {qa_pair.get('question')}")

# 使用示例
if __name__ == "__main__":
    # 文件路径
    dev_file_path = "/srv/nfs/home/njnu_zrq/RankCoT/src/data/test_data/asqa_dev.jsonl"
    query_to_answer_file_path = "/srv/nfs/home/njnu_zrq/RankCoT/src/answer_generation_new/data/asqa_queryCoT_to_answer.jsonl"
    output_file_path = "/srv/nfs/home/njnu_zrq/RankCoT/src/answer_generation_new/data/asqa_queryCoT_to_answer.jsonl"
    
    # 执行合并
    merge_asqa_data(dev_file_path, query_to_answer_file_path, output_file_path)
    
    # 验证结果
    verify_merge(output_file_path)
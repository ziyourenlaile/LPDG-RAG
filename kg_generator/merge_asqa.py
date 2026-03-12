import json

def merge_asqa_data_json(dev_file_path, query_to_answer_file_path, output_file_path):
    """
    将asqa_dev.jsonl中的qa_pairs内容添加到asqa_answer_updated.json中
    """
    
    # 1. 读取asqa_dev.jsonl数据，按sample_id组织
    dev_data = {}
    with open(dev_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            sample_id = data.get('sample_id')
            if sample_id:
                dev_data[sample_id] = data
    
    print(f"从asqa_dev.jsonl读取了 {len(dev_data)} 条数据")
    
    # 2. 读取asqa_answer_updated.json数据
    with open(query_to_answer_file_path, 'r', encoding='utf-8') as f:
        query_data = json.load(f)
    
    print(f"从asqa_answer_updated.json读取了数据")
    
    # 3. 处理数据：如果是列表格式
    if isinstance(query_data, list):
        updated_data = []
        for item in query_data:
            question_id = item.get('question_id')
            
            # 如果找到对应的dev数据，添加qa_pairs信息
            if question_id in dev_data:
                dev_item = dev_data[question_id]
                item['qa_pairs'] = dev_item.get('qa_pairs', [])
            
            updated_data.append(item)
        
        print(f"处理了 {len(updated_data)} 条数据（列表格式）")
        
    # 4. 处理数据：如果是字典格式（单条数据）
    elif isinstance(query_data, dict):
        question_id = query_data.get('question_id')
        
        # 如果找到对应的dev数据，添加qa_pairs信息
        if question_id in dev_data:
            dev_item = dev_data[question_id]
            query_data['qa_pairs'] = dev_item.get('qa_pairs', [])
        
        updated_data = query_data
        print(f"处理了单条数据（字典格式）")
    
    else:
        print("错误：asqa_answer_updated.json格式不支持")
        return
    
    # 5. 保存更新后的数据为JSON格式
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)
    
    print(f"合并后的数据已保存到: {output_file_path}")
    
    # 6. 统计信息
    if isinstance(updated_data, list):
        matched_count = sum(1 for item in updated_data if 'qa_pairs' in item)
        print(f"成功匹配并添加qa_pairs的数据: {matched_count} 条")
    else:
        if 'qa_pairs' in updated_data:
            print(f"成功添加qa_pairs，包含 {len(updated_data['qa_pairs'])} 个QA对")

def verify_merge_json(output_file_path):
    """
    验证合并结果
    """
    print("\n验证合并结果:")
    with open(output_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # 列表格式
        print(f"数据格式: 列表，包含 {len(data)} 个元素")
        for i, item in enumerate(data[:2]):  # 只显示前2个
            print(f"\n样本 {i+1}:")
            print(f"  question_id: {item.get('question_id')}")
            print(f"  question: {item.get('question')}")
            print(f"  是否包含qa_pairs: {'qa_pairs' in item}")
            if 'qa_pairs' in item:
                print(f"  qa_pairs数量: {len(item['qa_pairs'])}")
                for j, qa_pair in enumerate(item['qa_pairs'][:2]):  # 只显示前2个QA对
                    print(f"    QA对 {j+1}: {qa_pair.get('question')}")
    else:
        # 字典格式
        print(f"数据格式: 字典")
        print(f"  question_id: {data.get('question_id')}")
        print(f"  question: {data.get('question')}")
        print(f"  是否包含qa_pairs: {'qa_pairs' in data}")
        if 'qa_pairs' in data:
            print(f"  qa_pairs数量: {len(data['qa_pairs'])}")
            for j, qa_pair in enumerate(data['qa_pairs'][:3]):  # 显示前3个QA对
                print(f"    QA对 {j+1}: {qa_pair.get('question')}")

# 使用示例
if __name__ == "__main__":
    # 文件路径
    dev_file_path = "/srv/nfs/home/njnu_zrq/RankCoT/src/data/test_data/asqa_dev.jsonl"
    query_to_answer_file_path = "/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/grpo_1B_xiangxiguize/kg_passage_to_answer/asqa_answer.json"
    output_file_path = "/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/grpo_1B_xiangxiguize/kg_passage_to_answer/asqa_answer.json"
    
    # 执行合并
    merge_asqa_data_json(dev_file_path, query_to_answer_file_path, output_file_path)
    
    # 验证结果
    verify_merge_json(output_file_path)
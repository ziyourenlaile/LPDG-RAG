import json
import os
import argparse

def filter_kg_triples(input_path, output_path):
    """
    过滤知识图谱三元组数据，只保留符合数量要求的数据
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    
    # 读取数据
    print("正在读取数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据条数: {len(data)}")
    
    # 过滤数据
    filtered_data = []
    removed_count = 0
    
    for item in data:
        few_count = len(item.get('triples_few', []))
        medium_count = len(item.get('triples_medium', []))
        many_count = len(item.get('triples_many', []))
        
        # 检查数量是否符合要求
        if (1 <= few_count <= 3 and 
            4 <= medium_count <= 6 and 
            7 <= many_count <= 10):
            filtered_data.append(item)
        else:
            removed_count += 1
            print(f"删除数据 - passage_id: {item['passage_id']}, "
                  f"few: {few_count}, medium: {medium_count}, many: {many_count}")
    
    # 保存过滤后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 过滤完成 ===")
    print(f"原始数据条数: {len(data)}")
    print(f"过滤后数据条数: {len(filtered_data)}")
    print(f"删除数据条数: {removed_count}")
    print(f"保留比例: {len(filtered_data)/len(data)*100:.2f}%")
    print(f"结果保存至: {output_path}")
    
    # 统计过滤后的三元组数量分布
    if filtered_data:
        few_counts = [len(item['triples_few']) for item in filtered_data]
        medium_counts = [len(item['triples_medium']) for item in filtered_data]
        many_counts = [len(item['triples_many']) for item in filtered_data]
        
        print(f"\n=== 过滤后统计 ===")
        print(f"few三元组范围: {min(few_counts)}-{max(few_counts)}")
        print(f"medium三元组范围: {min(medium_counts)}-{max(medium_counts)}")
        print(f"many三元组范围: {min(many_counts)}-{max(many_counts)}")
        print(f"平均few三元组数: {sum(few_counts)/len(few_counts):.2f}")
        print(f"平均medium三元组数: {sum(medium_counts)/len(medium_counts):.2f}")
        print(f"平均many三元组数: {sum(many_counts)/len(many_counts):.2f}")

def main():
    parser = argparse.ArgumentParser(description='过滤知识图谱三元组数据')
    parser.add_argument('--input_path', type=str, 
                       default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples.json',
                       help='输入文件路径')
    parser.add_argument('--output_path', type=str, 
                       default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples_filtered.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input_path):
        print(f"错误: 输入文件不存在 - {args.input_path}")
        return
    
    filter_kg_triples(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
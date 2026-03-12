# import json
# import os
# import argparse
# from collections import defaultdict

# def merge_kg_files(few_path, medium_path, many_path, output_path):
#     """
#     合并三个知识图谱文件（按照passage_id进行合并）
    
#     Args:
#         few_path: few级别文件路径
#         medium_path: medium级别文件路径  
#         many_path: many级别文件路径
#         output_path: 合并后的输出路径
#     """
    
#     # 读取三个文件
#     print("正在读取文件...")
#     with open(few_path, 'r', encoding='utf-8') as f:
#         few_data = json.load(f)
    
#     with open(medium_path, 'r', encoding='utf-8') as f:
#         medium_data = json.load(f)
    
#     with open(many_path, 'r', encoding='utf-8') as f:
#         many_data = json.load(f)
    
#     print(f"few数据条数: {len(few_data)}")
#     print(f"medium数据条数: {len(medium_data)}")
#     print(f"many数据条数: {len(many_data)}")
    
#     # 创建以passage_id为键的字典
#     few_dict = {item['passage_id']: item for item in few_data}
#     medium_dict = {item['passage_id']: item for item in medium_data}
#     many_dict = {item['passage_id']: item for item in many_data}
    
#     # 获取所有唯一的passage_id
#     all_passage_ids = set(few_dict.keys()) | set(medium_dict.keys()) | set(many_dict.keys())
#     print(f"唯一passage_id数量: {len(all_passage_ids)}")
    
#     # 检查缺失的数据
#     missing_in_few = set(medium_dict.keys()) - set(few_dict.keys())
#     missing_in_medium = set(few_dict.keys()) - set(medium_dict.keys())
#     missing_in_many = set(few_dict.keys()) - set(many_dict.keys())
    
#     if missing_in_few:
#         print(f"警告: {len(missing_in_few)} 条数据在few文件中缺失")
#     if missing_in_medium:
#         print(f"警告: {len(missing_in_medium)} 条数据在medium文件中缺失")
#     if missing_in_many:
#         print(f"警告: {len(missing_in_many)} 条数据在many文件中缺失")
    
#     # 按照passage_id合并数据
#     merged_data = []
#     missing_count = 0
    
#     for passage_id in sorted(all_passage_ids):
#         # 检查每个文件是否包含该passage_id
#         if passage_id in few_dict and passage_id in medium_dict and passage_id in many_dict:
#             few_item = few_dict[passage_id]
#             medium_item = medium_dict[passage_id]
#             many_item = many_dict[passage_id]
            
#             # 验证基本信息是否一致
#             if (few_item['question_id'] != medium_item['question_id'] or 
#                 few_item['question_id'] != many_item['question_id']):
#                 print(f"警告: passage_id {passage_id} 的question_id不匹配")
#                 print(f"  few: {few_item['question_id']}, medium: {medium_item['question_id']}, many: {many_item['question_id']}")
            
#             # 合并数据
#             merged_item = {
#                 "question_id": few_item['question_id'],
#                 "question": few_item['question'],
#                 "answer": few_item['answer'],
#                 "passage_id": passage_id,
#                 "title": few_item['title'],
#                 "segment": few_item['segment'],
#                 "is_passage": few_item['is_passage'],
#                 "triples_few": few_item['triples_few'],
#                 "triples_medium": medium_item['triples_medium'],
#                 "triples_many": many_item['triples_many']
#             }
            
#             merged_data.append(merged_item)
#         else:
#             missing_count += 1
#             print(f"警告: passage_id {passage_id} 在部分文件中缺失，跳过合并")
    
#     # 保存合并后的数据
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
#     print(f"合并完成！共 {len(merged_data)} 条数据")
#     print(f"跳过 {missing_count} 条不完整数据")
#     print(f"结果保存至: {output_path}")
    
#     # 统计信息
#     total_few_triples = sum(len(item['triples_few']) for item in merged_data)
#     total_medium_triples = sum(len(item['triples_medium']) for item in merged_data)
#     total_many_triples = sum(len(item['triples_many']) for item in merged_data)
    
#     print(f"\n=== 合并统计 ===")
#     print(f"总数据条数: {len(merged_data)}")
#     print(f"few三元组总数: {total_few_triples}")
#     print(f"medium三元组总数: {total_medium_triples}")
#     print(f"many三元组总数: {total_many_triples}")
#     print(f"平均few三元组数: {total_few_triples/len(merged_data):.2f}")
#     print(f"平均medium三元组数: {total_medium_triples/len(merged_data):.2f}")
#     print(f"平均many三元组数: {total_many_triples/len(merged_data):.2f}")

# def main():
#     parser = argparse.ArgumentParser(description='合并知识图谱文件')
#     parser.add_argument('--few_path', type=str, help='few级别文件路径', default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples_few.json')
#     parser.add_argument('--medium_path', type=str, help='medium级别文件路径', default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples_medium.json')
#     parser.add_argument('--many_path', type=str, help='many级别文件路径', default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples_many.json')
#     parser.add_argument('--output_path', type=str, help='合并后的输出路径', default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples.json')
    
#     args = parser.parse_args()
    
#     # 检查文件是否存在
#     for path in [args.few_path, args.medium_path, args.many_path]:
#         if not os.path.exists(path):
#             print(f"错误: 文件不存在 - {path}")
#             return
    
#     merge_kg_files(args.few_path, args.medium_path, args.many_path, args.output_path)

# if __name__ == "__main__":
#     main()

import json
import os
import argparse
from collections import defaultdict

def merge_kg_files(few_path, medium_path, many_path, output_path):
    """
    合并三个知识图谱文件（按照question_id和passage_id进行合并）
    
    Args:
        few_path: few级别文件路径
        medium_path: medium级别文件路径  
        many_path: many级别文件路径
        output_path: 合并后的输出路径
    """
    
    # 读取三个文件
    print("正在读取文件...")
    with open(few_path, 'r', encoding='utf-8') as f:
        few_data = json.load(f)
    
    with open(medium_path, 'r', encoding='utf-8') as f:
        medium_data = json.load(f)
    
    with open(many_path, 'r', encoding='utf-8') as f:
        many_data = json.load(f)
    
    print(f"few数据条数: {len(few_data)}")
    print(f"medium数据条数: {len(medium_data)}")
    print(f"many数据条数: {len(many_data)}")
    
    # 创建复合键的字典：(question_id, passage_id) -> item
    def create_composite_dict(data):
        return {(item['question_id'], item['passage_id']): item for item in data}
    
    few_dict = create_composite_dict(few_data)
    medium_dict = create_composite_dict(medium_data)
    many_dict = create_composite_dict(many_data)
    
    # 获取所有唯一的复合键
    all_keys = set(few_dict.keys()) | set(medium_dict.keys()) | set(many_dict.keys())
    print(f"唯一(question_id, passage_id)组合数量: {len(all_keys)}")
    
    # 检查缺失的数据
    missing_in_few = set(medium_dict.keys()) - set(few_dict.keys())
    missing_in_medium = set(few_dict.keys()) - set(medium_dict.keys())
    missing_in_many = set(few_dict.keys()) - set(many_dict.keys())
    
    if missing_in_few:
        print(f"警告: {len(missing_in_few)} 条数据在few文件中缺失")
    if missing_in_medium:
        print(f"警告: {len(missing_in_medium)} 条数据在medium文件中缺失")
    if missing_in_many:
        print(f"警告: {len(missing_in_many)} 条数据在many文件中缺失")
    
    # 按照复合键合并数据
    merged_data = []
    missing_count = 0
    validation_errors = 0
    
    # 按question_id分组排序，便于查看
    sorted_keys = sorted(all_keys, key=lambda x: (x[0], x[1]))
    
    for key in sorted_keys:
        question_id, passage_id = key
        
        # 检查每个文件是否包含该复合键
        if key in few_dict and key in medium_dict and key in many_dict:
            few_item = few_dict[key]
            medium_item = medium_dict[key]
            many_item = many_dict[key]
            
            # 验证基本信息是否一致（双重保险）
            base_fields = ['question', 'answer', 'title', 'segment', 'is_passage']
            consistent = True
            
            for field in base_fields:
                if (few_item.get(field) != medium_item.get(field) or 
                    few_item.get(field) != many_item.get(field)):
                    print(f"警告: {question_id}-{passage_id} 的{field}字段不匹配")
                    print(f"  few: {few_item.get(field)}, medium: {medium_item.get(field)}, many: {many_item.get(field)}")
                    validation_errors += 1
                    consistent = False
            
            # 合并数据
            merged_item = {
                "question_id": question_id,
                "question": few_item['question'],
                "answer": few_item['answer'],
                "passage_id": passage_id,
                "title": few_item['title'],
                "segment": few_item['segment'],
                "is_passage": few_item['is_passage'],
                "triples_few": few_item['triples_few'],
                "triples_medium": medium_item['triples_medium'],
                "triples_many": many_item['triples_many']
            }
            
            merged_data.append(merged_item)
        else:
            missing_count += 1
            missing_sources = []
            if key not in few_dict: missing_sources.append("few")
            if key not in medium_dict: missing_sources.append("medium")
            if key not in many_dict: missing_sources.append("many")
            print(f"警告: {question_id}-{passage_id} 在{', '.join(missing_sources)}文件中缺失，跳过合并")
    
    # 保存合并后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 合并完成 ===")
    print(f"成功合并: {len(merged_data)} 条数据")
    print(f"跳过的不完整数据: {missing_count} 条")
    print(f"验证错误: {validation_errors} 处")
    print(f"结果保存至: {output_path}")
    
    # 详细统计信息
    total_few_triples = sum(len(item['triples_few']) for item in merged_data)
    total_medium_triples = sum(len(item['triples_medium']) for item in merged_data)
    total_many_triples = sum(len(item['triples_many']) for item in merged_data)
    
    print(f"\n=== 详细统计 ===")
    print(f"总数据条数: {len(merged_data)}")
    print(f"few三元组总数: {total_few_triples}")
    print(f"medium三元组总数: {total_medium_triples}")
    print(f"many三元组总数: {total_many_triples}")
    print(f"平均few三元组数: {total_few_triples/len(merged_data):.2f}")
    print(f"平均medium三元组数: {total_medium_triples/len(merged_data):.2f}")
    print(f"平均many三元组数: {total_many_triples/len(merged_data):.2f}")
    
    # 按question_id分组统计
    question_groups = defaultdict(list)
    for item in merged_data:
        question_groups[item['question_id']].append(item)
    
    print(f"\n=== 题目分布统计 ===")
    print(f"唯一题目数量: {len(question_groups)}")
    print(f"平均每个题目的passage数量: {len(merged_data)/len(question_groups):.2f}")

def main():
    parser = argparse.ArgumentParser(description='合并知识图谱文件')
    parser.add_argument('--few_path', type=str, help='few级别文件路径', 
                       default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples_few.json')
    parser.add_argument('--medium_path', type=str, help='medium级别文件路径', 
                       default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples_medium.json')
    parser.add_argument('--many_path', type=str, help='many级别文件路径', 
                       default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples_many.json')
    parser.add_argument('--output_path', type=str, help='合并后的输出路径', 
                       default='/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples.json')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for path_name, path in [('few', args.few_path), ('medium', args.medium_path), ('many', args.many_path)]:
        if not os.path.exists(path):
            print(f"错误: {path_name}文件不存在 - {path}")
            return
    
    merge_kg_files(args.few_path, args.medium_path, args.many_path, args.output_path)

if __name__ == "__main__":
    main()
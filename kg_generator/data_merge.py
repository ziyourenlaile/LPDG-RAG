import json
import jsonlines

# 读取原始答案文件
with open('/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/grpo_1B_xiangxiguize/kg_passage_to_answer/tqa_answer.json', 'r', encoding='utf-8') as f:
    answer_data = json.load(f)

# 创建question_id到ground_truth的映射
question_to_ground_truth = {}

# 从jsonl文件中提取ground_truth
with jsonlines.open('/srv/nfs/home/njnu_zrq/RankCoT/src/answer_generation/data/grpo_1B_xiangxiguize_2/query_to_cot/tqa_querypassage_to_CoT.jsonl', 'r') as reader:
    for item in reader:
        if 'id' in item and 'ground_truth' in item:
            question_to_ground_truth[item['id']] = item['ground_truth']

# 将ground_truth添加到答案数据中
for answer_item in answer_data:
    question_id = answer_item.get('question_id')
    if question_id in question_to_ground_truth:
        answer_item['ground_truth'] = question_to_ground_truth[question_id]
    else:
        # 如果没有找到对应的ground_truth，可以设置为空列表或None
        answer_item['ground_truth'] = []

# 保存更新后的文件
with open('/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/grpo_1B_xiangxiguize/kg_passage_to_answer/tqa_answer.json', 'w', encoding='utf-8') as f:
    json.dump(answer_data, f, indent=2, ensure_ascii=False)

print("处理完成！ground_truth字段已成功添加。")
print(f"总共处理了 {len(answer_data)} 条记录")
print(f"成功匹配到 {len([item for item in answer_data if item.get('ground_truth')])} 条记录的ground_truth")
# import json
# import jsonlines

# # 读取原始答案文件
# with open('/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/answer/kg_cot_short/tqa_answer.json', 'r', encoding='utf-8') as f:
#     answer_data = json.load(f)

# # 创建question_id到ground_truth的映射
# question_to_ground_truth = {}

# # 从jsonl文件中提取ground_truth
# with jsonlines.open('/srv/nfs/home/njnu_zrq/RankCoT/src/answer_generation/data/dpo/query_to_cot/tqa_querypassage_to_CoT.jsonl', 'r') as reader:
#     for item in reader:
#         if 'id' in item and 'ground_truth' in item:
#             question_to_ground_truth[item['id']] = item['ground_truth']

# # 将ground_truth添加到答案数据中
# for answer_item in answer_data:
#     question_id = answer_item.get('question_id')
#     if question_id in question_to_ground_truth:
#         answer_item['ground_truth'] = question_to_ground_truth[question_id]
#     else:
#         # 如果没有找到对应的ground_truth，可以设置为空列表或None
#         answer_item['ground_truth'] = []

# # 保存为JSONL文件
# output_path = '/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/answer/kg_cot_short/tqa_answer_merge.jsonl'
# with jsonlines.open(output_path, 'w') as writer:
#     for item in answer_data:
#         writer.write(item)

# print("处理完成！ground_truth字段已成功添加。")
# print(f"总共处理了 {len(answer_data)} 条记录")
# print(f"成功匹配到 {len([item for item in answer_data if item.get('ground_truth')])} 条记录的ground_truth")
# print(f"结果已保存为JSONL文件: {output_path}")
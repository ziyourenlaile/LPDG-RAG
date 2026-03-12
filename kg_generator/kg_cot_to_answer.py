import json
import argparse
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
from tqdm import tqdm

def load_and_validate_json(file_path):
    """加载并验证JSON文件，处理格式错误"""
    print(f"正在验证JSON文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✓ JSON文件格式正确")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON格式错误: {e}")
        raise

def load_cot_data(cot_file_path):
    """加载CoT数据"""
    print(f"正在加载CoT数据: {cot_file_path}")
    cot_data = {}
    
    with open(cot_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            question_id = data["id"]
            cot_data[question_id] = data["model_output"]
    
    print(f"✓ 成功加载 {len(cot_data)} 条CoT数据")
    return cot_data

class AnswerGenerator:
    def __init__(self, model_path, gpu="0"):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 修复vLLM配置 - 正确的参数设置
        self.llm = LLM(
            model=model_path,
            dtype='bfloat16',
            trust_remote_code=True,
            gpu_memory_utilization=0.85,  # 提高内存利用率
            # max_model_len=8192,           # 使用默认值或适当的值
            # enable_prefix_caching=False,  # 禁用前缀缓存以减少内存使用
            tensor_parallel_size=1,
            # enforce_eager=True,           # 禁用CUDA graphs避免断言错误
        )
        
        # 优化生成参数
        self.sampling_params = SamplingParams(
            n= 1,
            best_of= 1,
            presence_penalty= 1.0,
            frequency_penalty= 0.0,
            temperature= 1.0,
            top_p= 1.0,
            top_k= 1,
            use_beam_search= False,
            length_penalty= 1,
            early_stopping= False,
            stop= None,
            stop_token_ids= None,
            ignore_eos= False,
            max_tokens= 100,
            logprobs= None,
            prompt_logprobs= None,
            skip_special_tokens= True,
        )
        
        self.answer_instruction = (
            #### 1 asqa
            # "Based on the provided Chain of Thought reasoning and knowledge triples, generate a comprehensive and accurate answer to the question. "
            # "Use the Chain of Thought as the main reasoning process and supplement it with the structured knowledge from the triples. "
            # "If the provided information is insufficient, you may use your own knowledge to answer the question.\n"
            # "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n "
                
                        
            # #### 2
            # "Focus on the key information and provide a direct answer.\n\n"
            # "Requirements:\n"
            # "1. Be concise and to the point\n"
            # "2. Answer should be 1-3 sentences maximum\n"
            # "3. Use the most relevant information from both sources\n"
            # "4. Avoid unnecessary details and repetition\n\n"
            # "Format: A clear, brief paragraph that directly answers the question."
            
            # "Be concise and to the point\n"
            "Be concise , clear and brief\n"
            "Answer should be 3 sentences\n"
            "If the chain of thought and triples don't work, please answer the question based on your own knowledge.\n"
            "Please answer the question and only output the answer."
            
            # "Be concise and to the point\n"
            # "Answer should be 3 sentences\n"
            # # "Use the most relevant information from both sources\n"
            # # "Avoid unnecessary details and repetition\n\n"   # Knowledge Triples and 
            # "If the chain of thought don't work, please answer the question based on your own knowledge.\n"
            # "Please answer the question and only output the answer."            
            
            #### 3  rouge
            # "Read the given question and related Chain of Thought reasoning to gather relevant information.\n"
            # "The Chain of Thought reasoning provides the thinking process that should be used to answer the question.\n"
            # "Use the knowledge triples as supplementary structured information to support the reasoning.\n"
            # "If the provided information is insufficient, you may use your own knowledge to answer the question.\n"
            # "Please answer the question and only output a short ,clear and brief answer.\n"
        )

    def group_triples_by_question(self, triples_data, cot_data):
        """按问题ID分组三元组，并关联CoT数据"""
        question_groups = defaultdict(list)
        missing_cot_count = 0
        
        for item in triples_data:
            if not all(key in item for key in ['question_id', 'question', 'triples']):
                continue
                
            question_id = item["question_id"]
            
            # 获取对应的CoT数据
            cot_content = cot_data.get(question_id, "")
            if not cot_content:
                missing_cot_count += 1
                continue
            
            question_groups[question_id].append({
                "question": item["question"],
                "passage_id": item.get("passage_id", ""),
                "title": item.get("title", ""),
                "triples": item["triples"],
                "cot": cot_content  # 添加CoT内容
            })
        
        if missing_cot_count > 0:
            print(f"警告: 有 {missing_cot_count} 个问题缺少对应的CoT数据")
        
        return question_groups

    def create_answer_prompt(self, question, triples_group):
        """创建答案生成提示词，包含CoT和triples内容"""
        # 合并所有passage的三元组
        all_triples = []
        
        for passage_data in triples_group:
            # 处理三元组
            triples = passage_data["triples"]
            if isinstance(triples, list):
                all_triples.extend(triples)
        
        # 获取CoT内容（所有passage的CoT内容相同）
        cot_content = triples_group[0].get("cot", "") if triples_group else ""
        
        # 去重并过滤空三元组
        unique_triples = list(set([t for t in all_triples if t and isinstance(t, str)]))
        # unique_triples = unique_triples[:10]
        # 构建提示词内容
        triples_text = "\n".join(unique_triples) if unique_triples else "No triples available."
        
        cot_text = cot_content if cot_content else "No Chain of Thought reasoning available."
        
        
        prompt_content = f"Question: {question}\n\n"
        
        prompt_content += f"Chain of Thought:{cot_text}\n\n"
        
        prompt_content += f"Knowledge Triples:\n{triples_text}\n\n"
        
        # prompt_content += "Based on the above information, provide a comprehensive answer:"
        # prompt_content = self.answer_instruction + prompt_content
        prompt = [
            {"role": "system", "content": self.answer_instruction},
            {"role": "user", "content": prompt_content}
        ]
        
        input_prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        return input_prompt

    def generate_answers_batch(self, question_groups, batch_size=20):
        """分批生成答案，避免内存不足"""
        all_questions = list(question_groups.items())
        total_batches = (len(all_questions) + batch_size - 1) // batch_size
        
        all_results = []
        
        for batch_idx in tqdm(range(total_batches), desc="处理批次"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_questions))
            batch_questions = all_questions[start_idx:end_idx]
            
            prompts = []
            question_info = []
            
            for qid, triples_group in batch_questions:
                if not triples_group:
                    continue
                    
                question = triples_group[0]["question"]
                prompt = self.create_answer_prompt(question, triples_group)
                prompts.append(prompt)
                question_info.append({
                    "question_id": qid,
                    "question": question,
                    "passage_count": len(triples_group),
                    "triple_count": sum(len(p.get("triples", [])) for p in triples_group),
                    "has_cot": bool(triples_group[0].get("cot", ""))
                })
            
            if not prompts:
                continue
            
            try:
                # 使用vLLM生成答案
                outputs = self.llm.generate(prompts, self.sampling_params)
                
                # 提取生成的答案
                answers = [output.outputs[0].text.strip() for output in outputs]
                
                # 组装结果
                for i, answer in enumerate(answers):
                    all_results.append({
                        **question_info[i],
                        "model_answer": answer
                    })
                
            except Exception as e:
                print(f"批次 {batch_idx + 1} 处理失败: {e}")
                # 继续处理下一个批次
                continue
        
        return all_results

def generate_answers_from_triples(args):
    """从三元组和CoT生成答案的主函数"""
    # 读取并验证三元组数据
    try:
        triples_data = load_and_validate_json(args.triples_path)
    except Exception as e:
        print(f"无法加载三元组文件: {e}")
        return
    
    if not triples_data:
        print("没有找到有效数据")
        return
        
    print(f"成功加载 {len(triples_data)} 条三元组记录")
    
    # 加载CoT数据
    try:
        cot_data = load_cot_data(args.cot_path)
    except Exception as e:
        print(f"无法加载CoT文件: {e}")
        return
    
    # 初始化生成器
    answer_generator = AnswerGenerator(
        model_path=args.model_path,
        gpu=args.gpu
    )
    
    # 按问题分组并关联CoT数据
    question_groups = answer_generator.group_triples_by_question(triples_data, cot_data)
    print(f"找到 {len(question_groups)} 个唯一问题（包含CoT数据）")
    
    # 分批生成答案
    answers = answer_generator.generate_answers_batch(question_groups, batch_size=args.batch_size)
    
    # 保存结果
    if args.output_path:
        output_path = args.output_path
    else:
        base_name = os.path.splitext(args.triples_path)[0]
        output_path = f"{base_name}_cot_answers.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    print(f"答案已保存到: {output_path}")
    print(f"为 {len(answers)} 个问题生成了答案")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/mergemodel/mergemodel_output_dir/grpo")
    parser.add_argument('--triples_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/triples_fen_01/marco_triples.json")
    parser.add_argument('--cot_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/answer_generation/data/grpo_1B_only_AB/query_to_cot/marco_querypassage_to_CoT.jsonl", help="CoT数据文件路径")
    parser.add_argument('--output_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/grpo_1B_only_AB/marco_answer5.json")
    parser.add_argument('--gpu', type=str, default="5")
    parser.add_argument('--batch_size', type=int, default=60, help="批处理大小")
    args = parser.parse_args()
    
    generate_answers_from_triples(args)
# import json
# import argparse
# from collections import defaultdict
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer
# import os
# from tqdm import tqdm

# def load_and_validate_json(file_path):
#     """加载并验证JSON文件，处理格式错误"""
#     print(f"正在验证JSON文件: {file_path}")
    
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         print("✓ JSON文件格式正确")
#         return data
#     except json.JSONDecodeError as e:
#         print(f"JSON格式错误: {e}")
#         raise

# def load_cot_data(cot_file_path):
#     """加载CoT数据"""
#     print(f"正在加载CoT数据: {cot_file_path}")
#     cot_data = {}
    
#     with open(cot_file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line.strip())
#             question_id = data["id"]
#             cot_data[question_id] = data["model_output"]
    
#     print(f"✓ 成功加载 {len(cot_data)} 条CoT数据")
#     return cot_data

# class AnswerGenerator:
#     def __init__(self, model_path, gpu="0"):
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        
#         # 加载分词器
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path,
#             use_fast=True,
#             trust_remote_code=True,
#             padding_side="left",
#         )
        
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # 修复vLLM配置 - 正确的参数设置
#         self.llm = LLM(
#             model=model_path,
#             dtype='bfloat16',
#             trust_remote_code=True,
#             gpu_memory_utilization=0.85,
#             max_model_len=8192,
#             enable_prefix_caching=False,
#             tensor_parallel_size=1,
#             enforce_eager=True,
#         )
        
#         # 优化生成参数 - 减少答案长度
#         self.sampling_params = SamplingParams(
#             temperature=0.1,        # 降低温度，生成更确定的答案
#             top_p=0.8,              # 降低top_p，限制词汇选择
#             max_tokens=200,         # 显著减少最大生成长度
#             skip_special_tokens=True,
#             stop=[".\n", ".\n\n"],  # 添加停止词，遇到句号换行就停止
#             # n= 1,
#             # best_of= 1,
#             # presence_penalty= 1.0,
#             # frequency_penalty= 0.0,
#             # temperature= 1.0,
#             # top_p= 1.0,
#             # top_k= 1,
#             # use_beam_search= False,
#             # length_penalty= 1,
#             # early_stopping= False,
#             # stop= None,
#             # stop_token_ids= None,
#             # ignore_eos= False,
#             # max_tokens= 200,
#             # logprobs= None,
#             # prompt_logprobs= None,
#             # skip_special_tokens= True,
#         )
        
#         # 优化提示词 - 强调简洁性
#         self.answer_instruction = (
#             "Based on the provided Chain of Thought reasoning and knowledge triples, generate a concise and accurate answer to the question. "
#             "Focus on the key information and provide a direct answer.\n\n"
#             "Requirements:\n"
#             "1. Be concise and to the point\n"
#             "2. Answer should be 1-3 sentences maximum\n"
#             "3. Use the most relevant information from both sources\n"
#             "4. Avoid unnecessary details and repetition\n\n"
#             "Format: A clear, brief paragraph that directly answers the question."
#         )

#     def group_triples_by_question(self, triples_data, cot_data):
#         """按问题ID分组三元组，并关联CoT数据"""
#         question_groups = defaultdict(list)
#         missing_cot_count = 0
        
#         for item in triples_data:
#             if not all(key in item for key in ['question_id', 'question', 'triples']):
#                 continue
                
#             question_id = item["question_id"]
            
#             # 获取对应的CoT数据
#             cot_content = cot_data.get(question_id, "")
#             if not cot_content:
#                 missing_cot_count += 1
#                 continue
            
#             question_groups[question_id].append({
#                 "question": item["question"],
#                 "passage_id": item.get("passage_id", ""),
#                 "title": item.get("title", ""),
#                 "triples": item["triples"],
#                 "cot": cot_content
#             })
        
#         if missing_cot_count > 0:
#             print(f"警告: 有 {missing_cot_count} 个问题缺少对应的CoT数据")
        
#         return question_groups

#     def create_answer_prompt(self, question, triples_group):
#         """创建答案生成提示词，包含CoT和triples内容"""
#         # 合并所有passage的三元组
#         all_triples = []
        
#         for passage_data in triples_group:
#             triples = passage_data["triples"]
#             if isinstance(triples, list):
#                 all_triples.extend(triples)
        
#         # 获取CoT内容
#         cot_content = triples_group[0].get("cot", "") if triples_group else ""
        
#         # 去重并过滤空三元组
#         unique_triples = list(set([t for t in all_triples if t and isinstance(t, str)]))
        
#         # # 限制三元组数量，避免提示词过长
#         # if len(unique_triples) > 8:
#         #     unique_triples = unique_triples[:8]
        
#         # # 限制CoT长度
#         # if len(cot_content) > 300:
#         #     cot_content = cot_content[:300] + "..."
        
#         # 构建提示词内容
#         triples_text = "\n".join(unique_triples) if unique_triples else "No triples available."
        
#         cot_text = cot_content if cot_content else "No Chain of Thought reasoning available."
        
#         prompt_content = f"Question: {question}\n\n"
        
#         if cot_text != "No Chain of Thought reasoning available.":
#             prompt_content += f"Reasoning: {cot_text}\n\n"
        
#         if triples_text != "No triples available.":
#             prompt_content += f"Facts: {triples_text}\n\n"
        
#         # 简洁的提示
#         prompt_content += "Provide a concise answer:"
        
#         prompt = [
#             {"role": "system", "content": self.answer_instruction},
#             {"role": "user", "content": prompt_content}
#         ]
        
#         input_prompt = self.tokenizer.apply_chat_template(
#             prompt,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         return input_prompt

#     def generate_answers_batch(self, question_groups, batch_size=20):
#         """分批生成答案，避免内存不足"""
#         all_questions = list(question_groups.items())
#         total_batches = (len(all_questions) + batch_size - 1) // batch_size
        
#         all_results = []
        
#         for batch_idx in tqdm(range(total_batches), desc="处理批次"):
#             start_idx = batch_idx * batch_size
#             end_idx = min((batch_idx + 1) * batch_size, len(all_questions))
#             batch_questions = all_questions[start_idx:end_idx]
            
#             prompts = []
#             question_info = []
            
#             for qid, triples_group in batch_questions:
#                 if not triples_group:
#                     continue
                    
#                 question = triples_group[0]["question"]
#                 prompt = self.create_answer_prompt(question, triples_group)
#                 prompts.append(prompt)
#                 question_info.append({
#                     "question_id": qid,
#                     "question": question,
#                     "passage_count": len(triples_group),
#                     "triple_count": sum(len(p.get("triples", [])) for p in triples_group),
#                     "has_cot": bool(triples_group[0].get("cot", ""))
#                 })
            
#             if not prompts:
#                 continue
            
#             try:
#                 # 使用vLLM生成答案
#                 outputs = self.llm.generate(prompts, self.sampling_params)
                
#                 # 提取生成的答案
#                 answers = [output.outputs[0].text.strip() for output in outputs]
                
#                 # 组装结果
#                 for i, answer in enumerate(answers):
#                     all_results.append({
#                         **question_info[i],
#                         "model_answer": answer
#                     })
                
#             except Exception as e:
#                 print(f"批次 {batch_idx + 1} 处理失败: {e}")
#                 continue
        
#         return all_results

# def generate_answers_from_triples(args):
#     """从三元组和CoT生成答案的主函数"""
#     # 读取并验证三元组数据
#     try:
#         triples_data = load_and_validate_json(args.triples_path)
#     except Exception as e:
#         print(f"无法加载三元组文件: {e}")
#         return
    
#     if not triples_data:
#         print("没有找到有效数据")
#         return
        
#     print(f"成功加载 {len(triples_data)} 条三元组记录")
    
#     # 加载CoT数据
#     try:
#         cot_data = load_cot_data(args.cot_path)
#     except Exception as e:
#         print(f"无法加载CoT文件: {e}")
#         return
    
#     # 初始化生成器
#     answer_generator = AnswerGenerator(
#         model_path=args.model_path,
#         gpu=args.gpu
#     )
    
#     # 按问题分组并关联CoT数据
#     question_groups = answer_generator.group_triples_by_question(triples_data, cot_data)
#     print(f"找到 {len(question_groups)} 个唯一问题（包含CoT数据）")
    
#     # 分批生成答案
#     answers = answer_generator.generate_answers_batch(question_groups, batch_size=args.batch_size)
    
#     # 保存结果
#     if args.output_path:
#         output_path = args.output_path
#     else:
#         base_name = os.path.splitext(args.triples_path)[0]
#         output_path = f"{base_name}_cot_answers.json"
    
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(answers, f, ensure_ascii=False, indent=2)
    
#     print(f"答案已保存到: {output_path}")
#     print(f"为 {len(answers)} 个问题生成了答案")
    
#     # 显示答案长度统计
#     if answers:
#         avg_length = sum(len(answer["model_answer"]) for answer in answers) / len(answers)
#         max_length = max(len(answer["model_answer"]) for answer in answers)
#         min_length = min(len(answer["model_answer"]) for answer in answers)
#         print(f"答案长度统计 - 平均: {avg_length:.1f} 字符, 最长: {max_length}, 最短: {min_length}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct")
#     parser.add_argument('--triples_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/asqa_triples.json")
#     parser.add_argument('--cot_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/answer_generation/data/dpo/query_to_cot/asqa_querypassage_to_CoT.jsonl", help="CoT数据文件路径")
#     parser.add_argument('--output_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/answer/kg_cot_short_new/asqa_answer.json")
#     parser.add_argument('--gpu', type=str, default="2")
#     parser.add_argument('--batch_size', type=int, default=50, help="批处理大小")
#     args = parser.parse_args()
    
#     generate_answers_from_triples(args)
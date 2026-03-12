import os 
import json 
import argparse 
import numpy as np
from tqdm import tqdm 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# 提示词模板 - 简化版本，让模型基于自身知识回答
PROMPT_DICT = {
    "Mutichoice_query_to_answer": """
Please answer the following multiple choice question based on your own knowledge.
output only the correct choice letter between *****.

Question: {question}""",

    "QA_query_to_answer": """
Please answer the following question based on your own knowledge.
Provide a concise and accurate answer.

Question: {question}"""
}

def custom_json_decoder(obj):
    if 'id' in obj:
        obj['id'] = str(obj['id'])
    return obj

class llmDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def process_prompt(self, item):
        id = item['id']
        datatype = item['data_type']
        query = item['question']
        
        # 根据数据类型选择模板
        if datatype in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa']:
            template = PROMPT_DICT['Mutichoice_query_to_answer']
        elif datatype in ['gsm8k', 'strategyqa', 'web_questions', 'wiki_qa', 'yahoo_answers_qa', 'marcoqa']:
            template = PROMPT_DICT['QA_query_to_answer']
        else:
            # 默认使用QA模板
            template = PROMPT_DICT['QA_query_to_answer']
            
        template = template.format(question=query)
        messages = [
            {"role": "user", "content": template},
        ]
        input_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        item['input_prompt'] = input_prompt
        return item
    
    def __getitem__(self, index):
        item = self.data[index]       
        item = self.process_prompt(item)

        if index == 0:
            print("第一个样本的提示词:")
            print(item['input_prompt'])
            print("-" * 50)
       
        return item
    
    def __len__(self):
        return len(self.data)
    
    def Collactor(self, batch):
        return {
            'id': [f['id'] for f in batch],
            'data_type': [f['data_type'] for f in batch],
            'query': [f['question'] for f in batch],  # 修改为question字段
            'ground_truth': [f['answer'] for f in batch],
            'input_prompt': [f['input_prompt'] for f in batch]
        }

def merge_duplicate_questions(data):
    """合并重复的问题，只保留唯一的问题"""
    unique_questions = {}
    
    for item in data:
        question_id = item['id']
        if question_id not in unique_questions:
            # 只保留问题的基本信息，不包含passage
            unique_item = {
                'id': item['id'],
                'data_type': item['data_type'],
                'question': item['question'],
                'answer': item['answer']
            }
            unique_questions[question_id] = unique_item
    
    # 返回去重后的数据列表
    return list(unique_questions.values())

def inference(args):
    # 加载数据并合并重复问题
    with open(args.data_path, 'r') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
    
    print(f"原始数据条数: {len(data)}")
    
    # 合并重复问题
    merged_data = merge_duplicate_questions(data)
    print(f"去重后数据条数: {len(merged_data)}")
    
    # # 限制数据量用于测试
    # if args.test_mode:
    #     merged_data = merged_data[:args.test_num]
    #     print(f"测试模式: 使用前 {len(merged_data)} 条数据")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = llmDataset(merged_data, tokenizer)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        collate_fn=dataset.Collactor
    )
    
    # 配置采样参数
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": -1,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop": None,
        "stop_token_ids": None,
        "ignore_eos": False,
        "max_tokens": 128,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    
    # 配置GPU - 使用单GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cuda_num = 1  # 强制使用单GPU
    
    print(f"使用GPU: {args.gpus}, tensor_parallel_size: {cuda_num}")
    
    # 初始化vllm - 使用单GPU
    llm = LLM(
        model=args.model_path, 
        tensor_parallel_size=cuda_num, 
        dtype='bfloat16',
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_name), exist_ok=True)
    
    # 处理每个批次
    with open(args.output_name, 'w') as outfile:
        for batch in tqdm(dataloader, desc="处理进度"):
            input_prompt = batch['input_prompt']
            outputs = llm.generate(input_prompt, sampling_params)
            cleaned_outputs = [output.outputs[0].text for output in outputs]
            
            # 处理当前批次结果并保存
            for index in range(len(batch['id'])):
                output_item = {
                    "id": batch['id'][index],
                    "data_type": batch['data_type'][index],
                    "query": batch['query'][index],  # 保存为query字段
                    "model_answer": cleaned_outputs[index],
                    "ground_truth": batch['ground_truth'][index]  # 保存为ground_truth字段
                }
                json.dump(output_item, outfile)
                outfile.write('\n')
    
    print(f"推理完成，结果保存至: {args.output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/data/retriever_train_4000_noread_psg_modify10passage.jsonl")
    parser.add_argument('--output_name', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/CoTdata_generation/question_to_answer.jsonl")
    parser.add_argument('--gpus', type=str, default="0", help="指定要使用的单个GPU")
    parser.add_argument('--batch_size', type=int, default=64, help="每个批次的样本数量")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help="GPU内存利用率")
    # parser.add_argument('--test_mode', action='store_true', help="测试模式")
    # parser.add_argument('--test_num', type=int, default=10, help="测试模式下的数据数量")
    args = parser.parse_args()
    
    inference(args)
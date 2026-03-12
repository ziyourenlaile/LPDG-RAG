# import os 
# import json 
# import argparse 
# import numpy as np
# from tqdm import tqdm 
# from vllm import LLM, SamplingParams
# from template import PROMPT_DICT
# from transformers import AutoTokenizer
# from torch.utils.data import Dataset, DataLoader

# def custom_json_decoder(obj):
#     if 'id' in obj:
#         obj['id'] = str(obj['id'])
#     return obj

# class llmDataset(Dataset):
#     def __init__(self,data,tokenizer):
#         self.data = data
#         self.tokenizer = tokenizer
        
#     def process_prompt(self, item):
#         id=item['id']
#         datatype=item['data_type']
#         query = item['question']
#         passage = item['passage']['segment']
#         ground_truth = item['answer']
#         if datatype in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa']:
#             template = PROMPT_DICT['Mutichoice_querypassage_to_CoT']
#         if datatype in ['gsm8k', 'strategyqa', 'web_questions', 'wiki_qa', 'yahoo_answers_qa', 'marcoqa']:
#             template = PROMPT_DICT['QA_querypassage_to_CoT']
#         template = template.format(passage=passage, question=query)
#         messages = [
#             {"role": "user", "content": template},
#         ]
#         input_prompt = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         item['input_prompt'] = input_prompt
#         item['ground_truth'] = ground_truth
        

#         return item
    
            
#     def __getitem__(self, index):
#         item = self.data[index]       
#         item = self.process_prompt(item)

#         if index == 0:
#             print(item)
       
#         return item
    
#     def __len__(self):
#         return len(self.data)
    
#     def Collactor (self, batch):
        
#         id = [f['id'] for f in batch]
#         datatype = [f['data_type'] for f in batch]
#         query = [f['question'] for f in batch]
#         passage = [f['passage'] for f in batch]
#         ground_truth = [f['ground_truth'] for f in batch]
#         input_prompt = [f['input_prompt'] for f in batch]
        
#         return{ 'id':id,
#                'data_type':datatype,
#                'query':query,
#                'passage':passage,
#                'ground_truth':ground_truth,
#                'input_prompt': input_prompt
#         }
    
# def inference (args):
#     # Load data from the JSONL file
#     with open(args.data_path, 'r') as file:
#         data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
#     # with open(args.data_path, 'r') as file:
#     # # 读取前100行并解析
#     #     data = []
#     #     for i, line in enumerate(file):
#     #         if i < 10:  # 只处理前100行
#     #             data.append(json.loads(line, object_hook=custom_json_decoder))
#     #         else:
#     #             break  # 超过100行则停止
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#     dataset = llmDataset(data,tokenizer)
#     dataloader = DataLoader(dataset=dataset, batch_size=64, collate_fn= dataset.Collactor)
#     params_dict = {
#             "n": 1,
#             "best_of": 1,
#             "presence_penalty": 1.0,
#             "frequency_penalty": 0.0,
#             "temperature": 0.5,
#             "top_p": 0.8,
#             "top_k": -1,
#             "use_beam_search": False,
#             "length_penalty": 1,
#             "early_stopping": False,
#             "stop": None,
#             "stop_token_ids": None,
#             "ignore_eos": False,
#             "max_tokens": 512,
#             "logprobs": None,
#             "prompt_logprobs": None,
#             "skip_special_tokens": True,
#         }
#     sampling_params = SamplingParams(**params_dict)
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     cuda_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
#     # cuda_num = 1
#     llm = LLM(
#         model=args.model_path, 
#         tensor_parallel_size=cuda_num, 
#         dtype='bfloat16',
#         trust_remote_code=True,
#         gpu_memory_utilization=0.9
#     )
#     output_data = []
    
#     for batch in tqdm(dataloader):
#         input_prompt = batch['input_prompt']
#         outputs: list =llm.generate(input_prompt, sampling_params)
#         cleaned_outputs = [output.outputs[0].text for output in outputs]
#         maxindex = len(batch['id'])
#         for index in range(maxindex):
#             id=batch['id'][index]
#             datatype = batch['data_type'][index]
#             query = batch['query'][index]
#             passage = batch['passage'][index]
#             ground_truth = batch['ground_truth'][index]
#             model_output = cleaned_outputs[index] 
#             output_item = {
#                 "id":id,
#                 "data_type":datatype,
#                 "query": query,
#                 "model_output": model_output,
#                 "passage": passage,
#                 "ground_truth":ground_truth
#                 }
#             output_data.append(output_item)

#     with open(args.output_name, 'w') as outfile:
#         for item in output_data:
#             json.dump(item, outfile)
#             outfile.write('\n')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str,default="Meta-Llama-3-8B-Instruct")
#     parser.add_argument('--data_path',type=str, default="src/data/retriever_train_4000_noread_psg_modify10passage.jsonl")
#     parser.add_argument('--output_name',type=str,default="src/CoTdata_generation/querypassage_to_CoT.jsonl")
#     args = parser.parse_args ()
#     inference(args)
import os 
import json 
import argparse 
import numpy as np
from tqdm import tqdm 
from vllm import LLM, SamplingParams
from template import PROMPT_DICT
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def custom_json_decoder(obj):
    if 'id' in obj:
        obj['id'] = str(obj['id'])
    return obj

class llmDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def process_prompt(self, item):
        id=item['id']
        datatype=item['data_type']
        query = item['question']
        passage = item['passage']['segment']
        ground_truth = item['answer']
        if datatype in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa']:
            template = PROMPT_DICT['Mutichoice_querypassage_to_CoT']
        if datatype in ['gsm8k', 'strategyqa', 'web_questions', 'wiki_qa', 'yahoo_answers_qa', 'marcoqa']:
            template = PROMPT_DICT['QA_querypassage_to_CoT']
        template = template.format(passage=passage, question=query)
        messages = [
            {"role": "user", "content": template},
        ]
        input_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        item['input_prompt'] = input_prompt
        item['ground_truth'] = ground_truth
        

        return item
    
            
    def __getitem__(self, index):
        item = self.data[index]       
        item = self.process_prompt(item)

        if index == 0:
            print(item)
       
        return item
    
    def __len__(self):
        return len(self.data)
    
    def Collactor (self, batch):
        
        id = [f['id'] for f in batch]
        datatype = [f['data_type'] for f in batch]
        query = [f['question'] for f in batch]
        passage = [f['passage'] for f in batch]
        ground_truth = [f['ground_truth'] for f in batch]
        input_prompt = [f['input_prompt'] for f in batch]
        
        return{ 'id':id,
               'data_type':datatype,
               'query':query,
               'passage':passage,
               'ground_truth':ground_truth,
               'input_prompt': input_prompt
        }

def load_checkpoint(checkpoint_path):
    """加载检查点，返回已处理的样本ID集合"""
    processed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_ids.add(item['id'])
                except:
                    continue
    return processed_ids

def filter_unprocessed_data(data, processed_ids):
    """过滤掉已经处理过的数据"""
    return [item for item in data if item['id'] not in processed_ids]

def inference(args):
    # 加载原始数据
    with open(args.data_path, 'r') as file:
        all_data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
    
    # 加载已处理的样本ID（断点续传）
    processed_ids = load_checkpoint(args.output_name)
    print(f"已处理 {len(processed_ids)} 个样本，剩余 {len(all_data) - len(processed_ids)} 个样本待处理")
    
    # 过滤出未处理的数据
    data = filter_unprocessed_data(all_data, processed_ids)
    
    if not data:
        print("所有样本都已处理完毕！")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = llmDataset(data, tokenizer)
    # 使用与批处理大小相同的num_workers，避免数据加载成为瓶颈
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=64, 
        collate_fn=dataset.Collactor,
        num_workers=4
    )
    
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
        "max_tokens": 512,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
    llm = LLM(
        model=args.model_path, 
        tensor_parallel_size=cuda_num, 
        dtype='bfloat16',
        trust_remote_code=True,
        gpu_memory_utilization=0.9
    )
    
    # 处理每个批次并及时保存
    for batch in tqdm(dataloader, desc="处理进度"):
        input_prompt = batch['input_prompt']
        outputs: list = llm.generate(input_prompt, sampling_params)
        cleaned_outputs = [output.outputs[0].text for output in outputs]
        
        # 准备当前批次的输出数据
        batch_outputs = []
        maxindex = len(batch['id'])
        for index in range(maxindex):
            output_item = {
                "id": batch['id'][index],
                "data_type": batch['data_type'][index],
                "query": batch['query'][index],
                "model_output": cleaned_outputs[index],
                "passage": batch['passage'][index],
                "ground_truth": batch['ground_truth'][index]
            }
            batch_outputs.append(output_item)
        
        # 追加写入当前批次的结果（每64个样本保存一次）
        with open(args.output_name, 'a') as outfile:
            for item in batch_outputs:
                json.dump(item, outfile)
                outfile.write('\n')
        
        print(f"已完成 {len(batch_outputs)} 个样本处理，已保存到 {args.output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_path', type=str, default="src/data/retriever_train_4000_noread_psg_modify10passage.jsonl")
    parser.add_argument('--output_name', type=str, default="src/CoTdata_generation/query_to_CoT.jsonl")
    args = parser.parse_args()
    inference(args)

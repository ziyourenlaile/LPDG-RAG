# ## 原版
# import json
# import os
# import argparse
# from tqdm import tqdm
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer
# from torch.utils.data import Dataset, DataLoader

# def custom_json_decoder(obj):
#     if 'id' in obj:
#         obj['id'] = str(obj['id'])
#     return obj

# class CustomKGGenerator:
#     def __init__(self, model_path, max_length=4096, max_new_tokens=512, gpu="0"):
#         # 设置GPU
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu
#         cuda_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
        
#         # 加载分词器
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path,
#             use_fast=True,
#             trust_remote_code=True,
#             padding_side="left",
#             truncation_side="right",
#         )
        
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # 配置vLLM模型
#         self.llm = LLM(
#             model=model_path,
#             tensor_parallel_size=cuda_num,
#             dtype='bfloat16',
#             trust_remote_code=True,
#             gpu_memory_utilization=0.8
#         )
        
#         # 配置生成参数
#         self.sampling_params = SamplingParams(
#             n=1,
#             best_of=1,
#             presence_penalty=1.0,
#             frequency_penalty=0.0,
#             temperature=0.7,
#             top_p=1.0,
#             top_k=1,
#             use_beam_search=False,
#             length_penalty=1,
#             early_stopping=False,
#             stop=None,
#             stop_token_ids=None,
#             ignore_eos=False,
#             max_tokens=max_new_tokens,
#             skip_special_tokens=True,
#         )
        
#         # 更清晰的三元组提取任务指令
#         self.task_instruction = (
#             # "You are an expert at extracting structured knowledge from text. "
#             # "Your task is to extract knowledge triples in the format: <head entity; relation; tail entity>\n\n"
#             # "Rules:\n"
#             # "1. Each triple should represent a factual relationship\n"
#             # "2. Head and tail entities should be meaningful phrases from the text\n"
#             # "3. Relation should describe the relationship between head and tail\n"
#             # "4. If multiple tail entities share the same relation, combine them with commas\n"
#             # "5. Output ONLY the triples, one per line, in the exact format: <head; relation; tail>\n\n"
#             # "Examples:\n"
#             # "<Barack Obama; is president of; United States>\n"
#             # "<Python; is a; programming language>\n"
#             # "<Apple; manufactures; iPhone, iPad, MacBook>\n\n"
#             # "Now extract knowledge triples from the following document:"
#             """You are an expert at extracting knowledge triples. Please extract appropriate and key knowledge triples from the given text to help answer the question.

# Rules:
# 1. Extract an appropriate number of triples based on the content complexity and question requirements
# 2. Head and tail entities should be meaningful phrases from the text
# 3. Relations should clearly describe the relationship between head and tail entities
# 4. Output format must strictly follow: <head; relation; tail>
# 5. Focus on triples that help understand the core content and answer the question
# 6. For simple facts and straightforward questions, extract 1-3 triples (focus on core information)
# 7. For moderately complex topics, extract 4-6 triples (include key facts and some details)
# 8. For complex topics with rich information, extract 7-10 triples (comprehensive coverage)
# 9. Always prioritize the most important and relevant information
# 10. Adjust the number based on both text complexity and question specificity

# Please extract knowledge triples from the following question and text:"""
#         )

#     def process_document(self, doc):
#         """将文档转换为模型输入格式"""
#         title = doc.get("question", "")
#         segment = doc.get("segment", "")
#         # 构造文档文本（标题+内容）
#         doc_text = f"question: {title}\nText: {segment}"
#         return doc_text

#     def create_prompt(self, doc_text):
#         """创建对话格式的提示"""
#         prompt = [
#             {"role": "system", "content": self.task_instruction},
#             {"role": "user", "content": doc_text}
#         ]
        
#         # 应用聊天模板
#         input_prompt = self.tokenizer.apply_chat_template(
#             prompt,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         return input_prompt

#     def generate_triples_batch(self, doc_texts):
#         """批量生成知识三元组"""
#         # 创建提示
#         prompts = [self.create_prompt(doc_text) for doc_text in doc_texts]
        
#         # 使用vLLM生成
#         outputs = self.llm.generate(prompts, self.sampling_params)
        
#         # 提取生成的文本
#         generated_texts = [output.outputs[0].text for output in outputs]
        
#         # 解析三元组
#         all_triples = []
#         for generated_text in generated_texts:
#             triples = self._parse_triples(generated_text)
#             all_triples.append(triples)
        
#         return all_triples

#     def _parse_triples(self, generated_text):
#         """更灵活的三元组解析逻辑"""
#         triples = []
#         lines = generated_text.split("\n")
        
#         for line_num, line in enumerate(lines, 1):
#             line = line.strip()
#             if not line:
#                 continue
                
#             try:
#                 # 方法1: 严格匹配 <head; relation; tail> 格式
#                 if line.startswith("<") and line.endswith(">"):
#                     # 验证三元组格式
#                     content = line[1:-1].strip()
#                     if ";" in content:
#                         parts = [part.strip() for part in content.split(";")]
#                         if len(parts) >= 3:
#                             triples.append(line)
#                             continue
                
#                 # 方法2: 匹配包含分号的三元组格式（但不包含尖括号）
#                 if ";" in line and line.count(";") >= 2 and not (line.startswith("<") and line.endswith(">")):
#                     parts = [part.strip() for part in line.split(";")]
#                     if len(parts) >= 3:
#                         # 构建标准格式
#                         head = parts[0]
#                         relation = parts[1]
#                         tail = ";".join(parts[2:])  # 处理可能包含多个分号的tail
#                         standardized = f"<{head}; {relation}; {tail}>"
#                         triples.append(standardized)
#                         continue
                        
#                 # 方法3: 匹配其他分隔符格式
#                 if "->" in line:
#                     parts = line.split("->", 1)
#                     if len(parts) == 2:
#                         head = parts[0].strip()
#                         tail_part = parts[1].strip()
#                         # 尝试从tail中提取relation
#                         if ":" in tail_part:
#                             relation_parts = tail_part.split(":", 1)
#                             relation = relation_parts[0].strip()
#                             tail = relation_parts[1].strip()
#                         else:
#                             relation = "related to"
#                             tail = tail_part
#                         triples.append(f"<{head}; {relation}; {tail}>")
#                         continue
                        
#                 # 方法4: 匹配竖线分隔符
#                 if "|" in line and line.count("|") >= 2:
#                     parts = [part.strip() for part in line.split("|")]
#                     if len(parts) >= 3:
#                         triples.append(f"<{parts[0]}; {parts[1]}; {parts[2]}>")
#                         continue
                        
#             except Exception as e:
#                 # 如果解析过程中出现错误，记录警告但继续处理其他行
#                 print(f"警告: 解析第{line_num}行时出错: {line}")
#                 print(f"错误信息: {e}")
#                 continue
                
#         return triples

#     def save_triples(self, data_with_triples, save_path, use_indent=True):
#         """
#         保存包含三元组的数据到JSON文件
#         """
#         # 确保保存目录存在
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
#         # 保存数据
#         with open(save_path, "w", encoding="utf-8") as f:
#             if use_indent:
#                 json.dump(data_with_triples, f, ensure_ascii=False, indent=2)
#             else:
#                 json.dump(data_with_triples, f, ensure_ascii=False)
#         print(f"Triples saved to {save_path}")

# class KGDataset(Dataset):
#     def __init__(self, data, kg_generator, process_passages=True):
#         """
#         Args:
#             data: 原始数据列表
#             kg_generator: KG生成器实例
#             process_passages: 是否处理passages字段中的文档
#         """
#         self.data = data
#         self.kg_generator = kg_generator
#         self.process_passages = process_passages
        
#         # 预处理所有文档
#         self.processed_docs = self._preprocess_docs()
        
#     def _preprocess_docs(self):
#         """预处理所有文档"""
#         processed_docs = []
        
#         for item in self.data:
#             if self.process_passages and "passages" in item:
#                 # 处理每个passage作为单独的文档
#                 for passage in item["passages"]:
#                     processed_docs.append({
#                         "question_id": item.get("id", ""),
#                         "question": item.get("question", ""),
#                         "passage_id": passage.get("id", ""),
#                         "title": passage.get("title", ""),
#                         "segment": passage.get("segment", ""),
#                         "is_passage": True
#                     })
#             else:
#                 # 处理顶层文档（如果没有passages字段）
#                 processed_docs.append({
#                     "question_id": item.get("id", ""),
#                     "question": item.get("question", ""),
#                     "passage_id": item.get("id", ""),
#                     "title": item.get("title", ""),
#                     "segment": item.get("segment", ""),
#                     "is_passage": False
#                 })
        
#         print(f"Preprocessed {len(processed_docs)} documents from {len(self.data)} questions")
#         return processed_docs
        
#     def __getitem__(self, index):
#         item = self.processed_docs[index]
#         doc_text = self.kg_generator.process_document(item)
#         return {
#             "question_id": item["question_id"],
#             "question": item["question"],
#             "passage_id": item["passage_id"],
#             "title": item["title"],
#             "segment": item["segment"],
#             "is_passage": item["is_passage"],
#             "doc_text": doc_text
#         }
    
#     def __len__(self):
#         return len(self.processed_docs)
    
#     def collate_fn(self, batch):
#         return {
#             "question_ids": [f["question_id"] for f in batch],
#             "questions": [f["question"] for f in batch],
#             "passage_ids": [f["passage_id"] for f in batch],
#             "titles": [f["title"] for f in batch],
#             "segments": [f["segment"] for f in batch],
#             "is_passages": [f["is_passage"] for f in batch],
#             "doc_texts": [f["doc_text"] for f in batch]
#         }

# def generate_kg_triples(args):
#     """批量生成知识图谱三元组"""
#     # 读取数据
#     with open(args.data_path, 'r') as file:
#         data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
#     # data = data[:10]

#     print(f"Loaded {len(data)} questions")
    
#     # 初始化生成器
#     kg_generator = CustomKGGenerator(
#         model_path=args.model_path,
#         max_new_tokens=args.max_new_tokens,
#         gpu=args.gpu
#     )
    
#     # 创建数据集和数据加载器
#     dataset = KGDataset(data, kg_generator, process_passages=True)
#     dataloader = DataLoader(
#         dataset=dataset, 
#         batch_size=args.batch_size, 
#         collate_fn=dataset.collate_fn,
#         shuffle=False
#     )
    
#     # 处理每个批次并生成三元组
#     results = []
#     empty_count = 0
    
#     for batch in tqdm(dataloader, desc="Generating triples"):
#         doc_texts = batch["doc_texts"]
#         triples_batch = kg_generator.generate_triples_batch(doc_texts)
        
#         # 记录结果
#         for i in range(len(batch["passage_ids"])):
#             result = {
#                 "question_id": batch["question_ids"][i],
#                 "question": batch["questions"][i],
#                 "passage_id": batch["passage_ids"][i],
#                 "title": batch["titles"][i],
#                 "segment": batch["segments"][i],
#                 "is_passage": batch["is_passages"][i],
#                 "triples": triples_batch[i]
#             }
            
#             # 统计空结果
#             if not triples_batch[i]:
#                 empty_count += 1
                
#             results.append(result)
    
#     # 保存结果
#     kg_generator.save_triples(results, args.output_path)
    
#     # 输出统计信息
#     total_docs = len(results)
#     non_empty = total_docs - empty_count
#     print(f"\n=== 处理统计 ===")
#     print(f"总文档数: {total_docs}")
#     print(f"空三元组: {empty_count} ({empty_count/total_docs*100:.2f}%)")
#     print(f"非空三元组: {non_empty} ({non_empty/total_docs*100:.2f}%)")

# # 示例用法
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct")
#     parser.add_argument('--data_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/data/test_data/nq_dev_psg_modify10passage.jsonl")
#     parser.add_argument('--output_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/nq_triples.json")
#     parser.add_argument('--max_new_tokens', type=int, default=512)
#     parser.add_argument('--batch_size', type=int, default=70)
#     parser.add_argument('--gpu', type=str, default="1")
#     parser.add_argument('--process_passages', type=bool, default=True, help="Whether to process passages in the data")
#     args = parser.parse_args()
    
#     generate_kg_triples(args)

### 题目下的passage分开训练

## 原版
import json
import os
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def custom_json_decoder(obj):
    if 'id' in obj:
        obj['id'] = str(obj['id'])
    return obj

class CustomKGGenerator:
    def __init__(self, model_path, max_length=4096, max_new_tokens=512, gpu="0"):
        # 设置GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        cuda_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="right",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 配置vLLM模型
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=cuda_num,
            dtype='bfloat16',
            trust_remote_code=True,
            gpu_memory_utilization=0.8
        )
        
        # 配置生成参数
        self.sampling_params = SamplingParams(
            n=1,
            best_of=1,
            presence_penalty=1.0,
            frequency_penalty=0.0,
            temperature=0.7,
            top_p=1.0,
            top_k=1,
            use_beam_search=False,
            length_penalty=1,
            early_stopping=False,
            stop=None,
            stop_token_ids=None,
            ignore_eos=False,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        
        # 更清晰的三元组提取任务指令
        self.task_instruction = (
            """
            "You are an expert at extracting structured knowledge from text. "
            "Your task is to extract knowledge triples in the format: <head entity; relation; tail entity>\n\n"
            "Rules:\n"
            "1. Each triple should represent a factual relationship\n"
            "2. Head and tail entities should be meaningful phrases from the text\n"
            "3. Relation should describe the relationship between head and tail\n"
            "4. If multiple tail entities share the same relation, combine them with commas\n"
            "5. Output ONLY the triples, one per line, in the exact format: <head; relation; tail>\n"
            "6. Adjust the number of triples to generate based on the text complexity\n\n"
            "Examples:\n"
            "<Barack Obama; is president of; United States>\n"
            "<Python; is a; programming language>\n"
            "<Apple; manufactures; iPhone, iPad, MacBook>\n\n"
            "Now extract knowledge triples from the following document:"
            """
            # "You are an expert at extracting structured knowledge from text. "
            # "Your task is to extract knowledge triples in the format: <head entity; relation; tail entity>\n\n"
            # "Rules:\n"
            # "1. Each triple should represent a factual relationship\n"
            # "2. Head and tail entities should be meaningful phrases from the text\n"
            # "3. Relation should describe the relationship between head and tail\n"
            # "4. If multiple tail entities share the same relation, combine them with commas\n"
            # "5. Output ONLY the triples, one per line, in the exact format: <head; relation; tail>\n\n"
            # "Examples:\n"
            # "<Barack Obama; is president of; United States>\n"
            # "<Python; is a; programming language>\n"
            # "<Apple; manufactures; iPhone, iPad, MacBook>\n\n"
            # "Now extract knowledge triples from the following document:"
#             """You are an expert at extracting knowledge triples. Please extract appropriate and key knowledge triples from the given text to help answer the question.

# Rules:
# 1. Extract an appropriate number of triples based on the content complexity and question requirements
# 2. Head and tail entities should be meaningful phrases from the text
# 3. Relations should clearly describe the relationship between head and tail entities
# 4. Output format must strictly follow: <head; relation; tail>
# 5. Focus on triples that help understand the core content and answer the question
# 6. For simple facts and straightforward questions, extract 1-3 triples (focus on core information)
# 7. For moderately complex topics, extract 4-6 triples (include key facts and some details)
# 8. For complex topics with rich information, extract 7-10 triples (comprehensive coverage)
# 9. Always prioritize the most important and relevant information
# 10. Adjust the number based on both text complexity and question specificity

# Please extract knowledge triples from the following question and text:"""
        )

    def process_document(self, doc):
        """将文档转换为模型输入格式"""
        title = doc.get("question", "")
        segment = doc.get("segment", "")
        # 构造文档文本（标题+内容）
        doc_text = f"question: {title}\nText: {segment}"
        return doc_text

    def create_prompt(self, doc_text):
        """创建对话格式的提示"""
        prompt = [
            {"role": "system", "content": self.task_instruction},
            {"role": "user", "content": doc_text}
        ]
        
        # 应用聊天模板
        input_prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        return input_prompt

    def generate_triples_batch(self, doc_texts):
        """批量生成知识三元组"""
        # 创建提示
        prompts = [self.create_prompt(doc_text) for doc_text in doc_texts]
        
        # 使用vLLM生成
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # 提取生成的文本
        generated_texts = [output.outputs[0].text for output in outputs]
        
        # 解析三元组
        all_triples = []
        for generated_text in generated_texts:
            triples = self._parse_triples(generated_text)
            all_triples.append(triples)
        
        return all_triples

    def _parse_triples(self, generated_text):
        """更灵活的三元组解析逻辑"""
        triples = []
        lines = generated_text.split("\n")
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 方法1: 严格匹配 <head; relation; tail> 格式
                if line.startswith("<") and line.endswith(">"):
                    # 验证三元组格式
                    content = line[1:-1].strip()
                    if ";" in content:
                        parts = [part.strip() for part in content.split(";")]
                        if len(parts) >= 3:
                            triples.append(line)
                            continue
                
                # 方法2: 匹配包含分号的三元组格式（但不包含尖括号）
                if ";" in line and line.count(";") >= 2 and not (line.startswith("<") and line.endswith(">")):
                    parts = [part.strip() for part in line.split(";")]
                    if len(parts) >= 3:
                        # 构建标准格式
                        head = parts[0]
                        relation = parts[1]
                        tail = ";".join(parts[2:])  # 处理可能包含多个分号的tail
                        standardized = f"<{head}; {relation}; {tail}>"
                        triples.append(standardized)
                        continue
                        
                # 方法3: 匹配其他分隔符格式
                if "->" in line:
                    parts = line.split("->", 1)
                    if len(parts) == 2:
                        head = parts[0].strip()
                        tail_part = parts[1].strip()
                        # 尝试从tail中提取relation
                        if ":" in tail_part:
                            relation_parts = tail_part.split(":", 1)
                            relation = relation_parts[0].strip()
                            tail = relation_parts[1].strip()
                        else:
                            relation = "related to"
                            tail = tail_part
                        triples.append(f"<{head}; {relation}; {tail}>")
                        continue
                        
                # 方法4: 匹配竖线分隔符
                if "|" in line and line.count("|") >= 2:
                    parts = [part.strip() for part in line.split("|")]
                    if len(parts) >= 3:
                        triples.append(f"<{parts[0]}; {parts[1]}; {parts[2]}>")
                        continue
                        
            except Exception as e:
                # 如果解析过程中出现错误，记录警告但继续处理其他行
                print(f"警告: 解析第{line_num}行时出错: {line}")
                print(f"错误信息: {e}")
                continue
                
        return triples

    def save_triples(self, data_with_triples, save_path, use_indent=True):
        """
        保存包含三元组的数据到JSON文件
        """
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存数据
        with open(save_path, "w", encoding="utf-8") as f:
            if use_indent:
                json.dump(data_with_triples, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data_with_triples, f, ensure_ascii=False)
        print(f"Triples saved to {save_path}")

class KGDataset(Dataset):
    def __init__(self, data, kg_generator, process_passages=True):
        """
        Args:
            data: 原始数据列表
            kg_generator: KG生成器实例
            process_passages: 是否处理passages字段中的文档
        """
        self.data = data
        self.kg_generator = kg_generator
        self.process_passages = process_passages
        
        # 预处理所有文档
        self.processed_docs = self._preprocess_docs()
        
    def _preprocess_docs(self):
        """预处理所有文档"""
        processed_docs = []
        
        for item in self.data:
            if self.process_passages and "passages" in item:
                # 处理每个passage作为单独的文档
                for passage in item["passages"]:
                    processed_docs.append({
                        "question_id": item.get("id", ""),
                        "question": item.get("question", ""),
                        "passage_id": passage.get("id", ""),
                        "title": passage.get("title", ""),
                        "segment": passage.get("segment", ""),
                        "is_passage": True
                    })
            else:
                # 处理顶层文档（如果没有passages字段）
                processed_docs.append({
                    "question_id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "passage_id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "segment": item.get("segment", ""),
                    "is_passage": False
                })
        
        print(f"Preprocessed {len(processed_docs)} documents from {len(self.data)} questions")
        return processed_docs
        
    def __getitem__(self, index):
        item = self.processed_docs[index]
        doc_text = self.kg_generator.process_document(item)
        return {
            "question_id": item["question_id"],
            "question": item["question"],
            "passage_id": item["passage_id"],
            "title": item["title"],
            "segment": item["segment"],
            "is_passage": item["is_passage"],
            "doc_text": doc_text
        }
    
    def __len__(self):
        return len(self.processed_docs)
    
    def collate_fn(self, batch):
        return {
            "question_ids": [f["question_id"] for f in batch],
            "questions": [f["question"] for f in batch],
            "passage_ids": [f["passage_id"] for f in batch],
            "titles": [f["title"] for f in batch],
            "segments": [f["segment"] for f in batch],
            "is_passages": [f["is_passage"] for f in batch],
            "doc_texts": [f["doc_text"] for f in batch]
        }

def generate_kg_triples(args):
    """批量生成知识图谱三元组"""
    # 读取数据
    with open(args.data_path, 'r') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
    # data = data[:10]

    print(f"Loaded {len(data)} questions")
    
    # 初始化生成器
    kg_generator = CustomKGGenerator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        gpu=args.gpu
    )
    
    # 创建数据集和数据加载器
    dataset = KGDataset(data, kg_generator, process_passages=True)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        collate_fn=dataset.collate_fn,
        shuffle=False
    )
    
    # 处理每个批次并生成三元组
    results = []
    empty_count = 0
    
    for batch in tqdm(dataloader, desc="Generating triples"):
        doc_texts = batch["doc_texts"]
        triples_batch = kg_generator.generate_triples_batch(doc_texts)
        
        # 记录结果
        for i in range(len(batch["passage_ids"])):
            result = {
                "question_id": batch["question_ids"][i],
                "question": batch["questions"][i],
                "passage_id": batch["passage_ids"][i],
                "title": batch["titles"][i],
                "segment": batch["segments"][i],
                "is_passage": batch["is_passages"][i],
                "triples": triples_batch[i]
            }
            
            # 统计空结果
            if not triples_batch[i]:
                empty_count += 1
                
            results.append(result)
    
    # 保存结果
    kg_generator.save_triples(results, args.output_path)
    
    # 输出统计信息
    total_docs = len(results)
    non_empty = total_docs - empty_count
    print(f"\n=== 处理统计 ===")
    print(f"总文档数: {total_docs}")
    print(f"空三元组: {empty_count} ({empty_count/total_docs*100:.2f}%)")
    print(f"非空三元组: {non_empty} ({non_empty/total_docs*100:.2f}%)")

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/mergemodel/mergemodel_output_dir/kg")
    parser.add_argument('--data_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/data/test_data/asqa_dev_psg_modify10passage.jsonl")
    parser.add_argument('--output_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/triples_only_weitiao/asqa_triples.json")
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--process_passages', type=bool, default=True, help="Whether to process passages in the data")
    args = parser.parse_args()
    
    generate_kg_triples(args)

# ### 题目下的passage合并训练
# import json
# import os
# import argparse
# from tqdm import tqdm
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer
# from torch.utils.data import Dataset, DataLoader

# def custom_json_decoder(obj):
#     if 'id' in obj:
#         obj['id'] = str(obj['id'])
#     return obj

# class CustomKGGenerator:
#     def __init__(self, model_path, max_length=4096, max_new_tokens=512, gpu="0"):
#         # 设置GPU
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu
#         cuda_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
        
#         # 加载分词器
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path,
#             use_fast=True,
#             trust_remote_code=True,
#             padding_side="left",
#             truncation_side="right",
#         )
        
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # 配置vLLM模型
#         self.llm = LLM(
#             model=model_path,
#             tensor_parallel_size=cuda_num,
#             dtype='bfloat16',
#             trust_remote_code=True,
#             gpu_memory_utilization=0.8
#         )
        
#         # 配置生成参数
#         self.sampling_params = SamplingParams(
#             n=1,
#             best_of=1,
#             presence_penalty=1.0,
#             frequency_penalty=0.0,
#             temperature=0.7,
#             top_p=1.0,
#             top_k=1,
#             use_beam_search=False,
#             length_penalty=1,
#             early_stopping=False,
#             stop=None,
#             stop_token_ids=None,
#             ignore_eos=False,
#             max_tokens=max_new_tokens,
#             skip_special_tokens=True,
#         )
        
#         # 更清晰的三元组提取任务指令
#         self.task_instruction = """
#             "You are an expert at extracting structured knowledge from text. "
#             "Your task is to extract knowledge triples in the format: <head entity; relation; tail entity>\n\n"
#             "Rules:\n"
#             "1. Each triple should represent a factual relationship\n"
#             "2. Head and tail entities should be meaningful phrases from the text\n"
#             "3. Relation should describe the relationship between head and tail\n"
#             "4. If multiple tail entities share the same relation, combine them with commas\n"
#             "5. Output ONLY the triples, one per line, in the exact format: <head; relation; tail>\n"
#             "6. Adjust the number of triples to generate based on the text complexity\n\n"
#             "Examples:\n"
#             "<Barack Obama; is president of; United States>\n"
#             "<Python; is a; programming language>\n"
#             "<Apple; manufactures; iPhone, iPad, MacBook>\n\n"
#             "Now extract knowledge triples from the following document:"
#             """
# #         """You are an expert at extracting knowledge triples. Please extract appropriate and key knowledge triples from the given text to help answer the question.

# # Rules:
# # 1. Extract an appropriate number of triples based on the content complexity and question requirements
# # 2. Head and tail entities should be meaningful phrases from the text
# # 3. Relations should clearly describe the relationship between head and tail entities
# # 4. Output format must strictly follow: <head; relation; tail>
# # 5. Focus on triples that help understand the core content and answer the question
# # 6. For simple facts and straightforward questions, extract 1-3 triples (focus on core information)
# # 7. For moderately complex topics, extract 4-6 triples (include key facts and some details)
# # 8. For complex topics with rich information, extract 7-10 triples (comprehensive coverage)
# # 9. Always prioritize the most important and relevant information
# # 10. Adjust the number based on both text complexity and question specificity

# # Please extract knowledge triples from the following question and text:"""

#     def merge_segments_for_question(self, question_data, max_length=3000):
#         """将一个question下的所有segment合并，最大长度为max_length"""
#         question = question_data.get("question", "")
#         passages = question_data.get("passages", [])
        
#         if not passages:
#             # 如果没有passages，使用顶层字段
#             segment = question_data.get("segment", "")
#             title = question_data.get("title", "")
#             merged_text = f"{title}\n\n{segment}" if title else segment
#         else:
#             # 合并所有passages的segment
#             segments = []
#             current_length = 0
            
#             for passage in passages:
#                 title = passage.get("title", "")
#                 segment = passage.get("segment", "")
#                 passage_text = f"{title}\n\n{segment}" if title else segment
                
#                 # 检查是否超过最大长度
#                 if current_length + len(passage_text) <= max_length:
#                     segments.append(passage_text)
#                     current_length += len(passage_text)
#                 else:
#                     # 如果添加当前passage会超过长度，则停止
#                     remaining_space = max_length - current_length
#                     if remaining_space > 100:  # 至少保留100字符空间
#                         truncated_text = passage_text[:remaining_space] + "..."
#                         segments.append(truncated_text)
#                     break
        
#             merged_text = "\n\n".join(segments)
        
#         return {
#             "question_id": question_data.get("id", ""),
#             "question": question,
#             "merged_segment": merged_text,
#             "segment_count": len(passages) if passages else 1,
#             "total_length": len(merged_text)
#         }

#     def process_document(self, doc):
#         """将文档转换为模型输入格式"""
#         question = doc.get("question", "")
#         segment = doc.get("merged_segment", "")
#         # 构造文档文本
#         doc_text = f"Question: {question}\n\nText: {segment}"
#         return doc_text

#     def create_prompt(self, doc_text):
#         """创建对话格式的提示"""
#         prompt = [
#             {"role": "system", "content": self.task_instruction},
#             {"role": "user", "content": doc_text}
#         ]
        
#         # 应用聊天模板
#         input_prompt = self.tokenizer.apply_chat_template(
#             prompt,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         return input_prompt

#     def generate_triples_batch(self, doc_texts):
#         """批量生成知识三元组"""
#         # 创建提示
#         prompts = [self.create_prompt(doc_text) for doc_text in doc_texts]
        
#         # 使用vLLM生成
#         outputs = self.llm.generate(prompts, self.sampling_params)
        
#         # 提取生成的文本
#         generated_texts = [output.outputs[0].text for output in outputs]
        
#         # 解析三元组
#         all_triples = []
#         for generated_text in generated_texts:
#             triples = self._parse_triples(generated_text)
#             all_triples.append(triples)
        
#         return all_triples

#     def _parse_triples(self, generated_text):
#         """更灵活的三元组解析逻辑"""
#         triples = []
#         lines = generated_text.split("\n")
        
#         for line_num, line in enumerate(lines, 1):
#             line = line.strip()
#             if not line:
#                 continue
                
#             try:
#                 # 方法1: 严格匹配 <head; relation; tail> 格式
#                 if line.startswith("<") and line.endswith(">"):
#                     # 验证三元组格式
#                     content = line[1:-1].strip()
#                     if ";" in content:
#                         parts = [part.strip() for part in content.split(";")]
#                         if len(parts) >= 3:
#                             triples.append(line)
#                             continue
                
#                 # 方法2: 匹配包含分号的三元组格式（但不包含尖括号）
#                 if ";" in line and line.count(";") >= 2 and not (line.startswith("<") and line.endswith(">")):
#                     parts = [part.strip() for part in line.split(";")]
#                     if len(parts) >= 3:
#                         # 构建标准格式
#                         head = parts[0]
#                         relation = parts[1]
#                         tail = ";".join(parts[2:])  # 处理可能包含多个分号的tail
#                         standardized = f"<{head}; {relation}; {tail}>"
#                         triples.append(standardized)
#                         continue
                        
#                 # 方法3: 匹配其他分隔符格式
#                 if "->" in line:
#                     parts = line.split("->", 1)
#                     if len(parts) == 2:
#                         head = parts[0].strip()
#                         tail_part = parts[1].strip()
#                         # 尝试从tail中提取relation
#                         if ":" in tail_part:
#                             relation_parts = tail_part.split(":", 1)
#                             relation = relation_parts[0].strip()
#                             tail = relation_parts[1].strip()
#                         else:
#                             relation = "related to"
#                             tail = tail_part
#                         triples.append(f"<{head}; {relation}; {tail}>")
#                         continue
                        
#                 # 方法4: 匹配竖线分隔符
#                 if "|" in line and line.count("|") >= 2:
#                     parts = [part.strip() for part in line.split("|")]
#                     if len(parts) >= 3:
#                         triples.append(f"<{parts[0]}; {parts[1]}; {parts[2]}>")
#                         continue
                        
#             except Exception as e:
#                 # 如果解析过程中出现错误，记录警告但继续处理其他行
#                 print(f"警告: 解析第{line_num}行时出错: {line}")
#                 print(f"错误信息: {e}")
#                 continue
                
#         return triples

#     def save_triples(self, data_with_triples, save_path, use_indent=True):
#         """
#         保存包含三元组的数据到JSON文件
#         """
#         # 确保保存目录存在
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
#         # 保存数据
#         with open(save_path, "w", encoding="utf-8") as f:
#             if use_indent:
#                 json.dump(data_with_triples, f, ensure_ascii=False, indent=2)
#             else:
#                 json.dump(data_with_triples, f, ensure_ascii=False)
#         print(f"Triples saved to {save_path}")

# class KGDataset(Dataset):
#     def __init__(self, data, kg_generator, max_segment_length=3000):
#         """
#         Args:
#             data: 原始数据列表
#             kg_generator: KG生成器实例
#             max_segment_length: 合并segment的最大长度
#         """
#         self.data = data
#         self.kg_generator = kg_generator
#         self.max_segment_length = max_segment_length
        
#         # 预处理所有文档 - 合并每个question的segments
#         self.processed_docs = self._preprocess_docs()
        
#     def _preprocess_docs(self):
#         """预处理所有文档 - 合并每个question的segments"""
#         processed_docs = []
        
#         for item in self.data:
#             # 合并该question下的所有segments
#             merged_doc = self.kg_generator.merge_segments_for_question(
#                 item, 
#                 max_length=self.max_segment_length
#             )
#             processed_docs.append(merged_doc)
        
#         print(f"Preprocessed {len(processed_docs)} questions (merged segments)")
        
#         # 输出统计信息
#         total_segments = sum(doc.get("segment_count", 0) for doc in processed_docs)
#         avg_length = sum(doc.get("total_length", 0) for doc in processed_docs) / len(processed_docs)
#         print(f"Total original segments: {total_segments}")
#         print(f"Average merged text length: {avg_length:.0f} characters")
        
#         return processed_docs
        
#     def __getitem__(self, index):
#         item = self.processed_docs[index]
#         doc_text = self.kg_generator.process_document(item)
#         return {
#             "question_id": item["question_id"],
#             "question": item["question"],
#             "merged_segment": item["merged_segment"],
#             "segment_count": item["segment_count"],
#             "total_length": item["total_length"],
#             "doc_text": doc_text
#         }
    
#     def __len__(self):
#         return len(self.processed_docs)
    
#     def collate_fn(self, batch):
#         return {
#             "question_ids": [f["question_id"] for f in batch],
#             "questions": [f["question"] for f in batch],
#             "merged_segments": [f["merged_segment"] for f in batch],
#             "segment_counts": [f["segment_count"] for f in batch],
#             "total_lengths": [f["total_length"] for f in batch],
#             "doc_texts": [f["doc_text"] for f in batch]
#         }

# def generate_kg_triples(args):
#     """批量生成知识图谱三元组"""
#     # 读取数据
#     with open(args.data_path, 'r') as file:
#         data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
#     # data = data[:10]  # 用于测试

#     print(f"Loaded {len(data)} questions")
    
#     # 初始化生成器
#     kg_generator = CustomKGGenerator(
#         model_path=args.model_path,
#         max_new_tokens=args.max_new_tokens,
#         gpu=args.gpu
#     )
    
#     # 创建数据集和数据加载器
#     dataset = KGDataset(data, kg_generator, max_segment_length=args.max_segment_length)
#     dataloader = DataLoader(
#         dataset=dataset, 
#         batch_size=args.batch_size, 
#         collate_fn=dataset.collate_fn,
#         shuffle=False
#     )
    
#     # 处理每个批次并生成三元组
#     results = []
#     empty_count = 0
    
#     for batch in tqdm(dataloader, desc="Generating triples"):
#         doc_texts = batch["doc_texts"]
#         triples_batch = kg_generator.generate_triples_batch(doc_texts)
        
#         # 记录结果
#         for i in range(len(batch["question_ids"])):
#             result = {
#                 "question_id": batch["question_ids"][i],
#                 "question": batch["questions"][i],
#                 "merged_segment": batch["merged_segments"][i],
#                 "segment_count": batch["segment_counts"][i],
#                 "total_length": batch["total_lengths"][i],
#                 "triples": triples_batch[i]
#             }
            
#             # 统计空结果
#             if not triples_batch[i]:
#                 empty_count += 1
                
#             results.append(result)
    
#     # 保存结果
#     kg_generator.save_triples(results, args.output_path)
    
#     # 输出统计信息
#     total_docs = len(results)
#     non_empty = total_docs - empty_count
#     print(f"\n=== 处理统计 ===")
#     print(f"总问题数: {total_docs}")
#     print(f"空三元组: {empty_count} ({empty_count/total_docs*100:.2f}%)")
#     print(f"非空三元组: {non_empty} ({non_empty/total_docs*100:.2f}%)")
    
#     # 输出三元组数量统计
#     total_triples = sum(len(result["triples"]) for result in results)
#     avg_triples_per_question = total_triples / total_docs if total_docs > 0 else 0
#     print(f"总三元组数: {total_triples}")
#     print(f"平均每个问题三元组数: {avg_triples_per_question:.2f}")

# # 示例用法
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/mergemodel/mergemodel_output_dir/kg")
#     parser.add_argument('--data_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/data/test_data/tqa_dev_psg_modify10passage.jsonl")
#     parser.add_argument('--output_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/kg_generator/data/triples_all_02/tqa_triples.json")
#     parser.add_argument('--max_new_tokens', type=int, default=512)
#     parser.add_argument('--batch_size', type=int, default=50)
#     parser.add_argument('--gpu', type=str, default="7")
#     parser.add_argument('--max_segment_length', type=int, default=3000, help="合并segment的最大长度")
#     args = parser.parse_args()
    
#     generate_kg_triples(args)
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
    def __init__(self, model_path, max_length=4096, gpu="0"):
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
        
        # 为不同级别设置不同的max_new_tokens
        self.max_new_tokens_by_level = {
            "few": 256,      # 少量三元组，较短的输出
            "medium": 384,   # 中等数量三元组，中等长度输出
            "many": 512      # 大量三元组，较长输出
        }

        # 三种不同数量级别的指令
        self.task_instructions = {
            "few": (
                "You are an expert at extracting structured knowledge from text. "
                "Your task is to extract 1-3 most important knowledge triples in the format: <head entity; relation; tail entity>\n\n"
                "Rules:\n"
                "1. Extract ONLY the 1-3 most crucial and important triples\n"
                "2. Focus on the most salient facts and relationships\n"
                "3. Head and tail entities should be meaningful phrases from the text\n"
                "4. Relation should describe the relationship between head and tail\n"
                "5. Output ONLY the triples, one per line, in the exact format: <head; relation; tail>\n\n"
                "Examples:\n"
                "<Barack Obama; is president of; United States>\n"
                "<Python; is a; programming language>\n\n"
                "Now extract the most important knowledge triples from the following document:"
            ),
            "medium": (
                "You are an expert at extracting structured knowledge from text. "
                "Your task is to extract 4-6 key knowledge triples in the format: <head entity; relation; tail entity>\n\n"
                "Rules:\n"
                "1. Extract 4-6 important triples that capture key information\n"
                "2. Include both major facts and some supporting details\n"
                "3. Head and tail entities should be meaningful phrases from the text\n"
                "4. Relation should describe the relationship between head and tail\n"
                "5. If multiple tail entities share the same relation, combine them with commas\n"
                "6. Output ONLY the triples, one per line, in the exact format: <head; relation; tail>\n\n"
                "Examples:\n"
                "<Barack Obama; is president of; United States>\n"
                "<Python; is a; programming language>\n"
                "<Apple; manufactures; iPhone, iPad, MacBook>\n\n"
                "Now extract key knowledge triples from the following document:"
            ),
            "many": (
                "You are an expert at extracting structured knowledge from text. "
                "Your task is to extract knowledge triples in the format: <head entity; relation; tail entity>\n\n"
                "Rules:\n"
                "1. Each triple should represent a factual relationship\n"
                "2. Head and tail entities should be meaningful phrases from the text\n"
                "3. Relation should describe the relationship between head and tail\n"
                "4. If multiple tail entities share the same relation, combine them with commas\n"
                "5. Output ONLY the triples, one per line, in the exact format: <head; relation; tail>\n\n"
                "Examples:\n"
                "<Barack Obama; is president of; United States>\n"
                "<Python; is a; programming language>\n"
                "<Apple; manufactures; iPhone, iPad, MacBook>\n\n"
                "Now extract knowledge triples from the following document:"
                # "You are an expert at extracting structured knowledge from text. "
                # "Your task is to extract 7-10 comprehensive knowledge triples in the format: <head entity; relation; tail entity>\n\n"
                # "Rules:\n"
                # "1. Extract 7-10 comprehensive triples covering various aspects\n"
                # "2. Include major facts, supporting details, and contextual information\n"
                # "3. Head and tail entities should be meaningful phrases from the text\n"
                # "4. Relation should describe the relationship between head and tail\n"
                # "5. If multiple tail entities share the same relation, combine them with commas\n"
                # "6. Try to capture diverse types of relationships\n"
                # "7. Output ONLY the triples, one per line, in the exact format: <head; relation; tail>\n\n"
                # "Examples:\n"
                # "<Barack Obama; is president of; United States>\n"
                # "<Python; is a; programming language>\n"
                # "<Apple; manufactures; iPhone, iPad, MacBook>\n"
                # "<Einstein; developed; theory of relativity>\n\n"
                # "Now extract comprehensive knowledge triples from the following document:"
            )
        }

    def process_document(self, doc):
        """将文档转换为模型输入格式（不包含answer）"""
        question = doc.get("question", "")
        # title = doc.get("title", "")
        segment = doc.get("segment", "")
        
        # 构造文档文本（不包含answer）
        doc_parts = []
        if question:
            doc_parts.append(f"Question: {question}")
        # if title:
        #     doc_parts.append(f"Title: {title}")
        if segment:
            doc_parts.append(f"Text: {segment}")
        
        doc_text = "\n".join(doc_parts)
        return doc_text

    def create_prompt(self, doc_text, quantity_level):
        """创建对话格式的提示"""
        instruction = self.task_instructions[quantity_level]
        prompt = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": doc_text}
        ]
        
        # 应用聊天模板
        input_prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        return input_prompt

    def get_sampling_params(self, level):
        """根据级别获取对应的生成参数"""
        max_tokens = self.max_new_tokens_by_level[level]
        
        return SamplingParams(
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
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

    def generate_triples_for_level(self, doc_texts, level):
        """为每个文档生成指定级别的三元组"""
        print(f"Generating {level} triples (max_tokens: {self.max_new_tokens_by_level[level]})...")
        
        # 创建对应级别的提示
        prompts = [self.create_prompt(doc_text, level) for doc_text in doc_texts]
        
        # 使用对应级别的生成参数
        sampling_params = self.get_sampling_params(level)
        
        # 使用vLLM生成
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 提取生成的文本并解析三元组
        triples_list = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            triples = self._parse_triples(generated_text)
            triples_list.append(triples)
        
        return triples_list

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
                        "data_type" :item.get("data_type", ""),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),  # 提取answer但不用于生成
                        "passage_id": passage.get("id", ""),
                        "title": passage.get("title", ""),
                        "segment": passage.get("segment", ""),
                        "is_passage": True
                    })
            else:
                # 处理顶层文档（如果没有passages字段）
                # 检查是否有passage字段（单数形式）
                if "passage" in item and isinstance(item["passage"], dict):
                    passage = item["passage"]
                    processed_docs.append({
                        "question_id": item.get("id", ""),
                        "data_type" :item.get("data_type", ""),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),  # 提取answer但不用于生成
                        "passage_id": passage.get("id", ""),
                        "title": passage.get("title", ""),
                        "segment": passage.get("segment", ""),
                        "is_passage": True
                    })
                else:
                    # 处理没有passage字段的情况
                    processed_docs.append({
                        "question_id": item.get("id", ""),
                        "data_type" :item.get("data_type", ""),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),  # 提取answer但不用于生成
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
            "data_type" :item["data_type"],
            "question": item["question"],
            "answer": item["answer"],  # 提取answer但不用于生成
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
            "data_types" :[f["data_type"] for f in batch],
            "questions": [f["question"] for f in batch],
            "answers": [f["answer"] for f in batch],
            "passage_ids": [f["passage_id"] for f in batch],
            "titles": [f["title"] for f in batch],
            "segments": [f["segment"] for f in batch],
            "is_passages": [f["is_passage"] for f in batch],
            "doc_texts": [f["doc_text"] for f in batch]
        }

def generate_kg_triples_for_level(args, level):
    """为指定级别批量生成知识图谱三元组"""
    # 读取数据
    with open(args.data_path, 'r') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
    # data = data[:50]  # 用于测试的小批量数据

    print(f"Loaded {len(data)} questions")
    
    # 初始化生成器
    kg_generator = CustomKGGenerator(
        model_path=args.model_path,
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
    
    # 处理每个批次并生成指定级别的三元组
    results = []
    empty_count = 0
    triple_counts = []
    
    for batch in tqdm(dataloader, desc=f"Generating {level} triples"):
        doc_texts = batch["doc_texts"]
        
        # 生成指定级别的三元组
        triples_list = kg_generator.generate_triples_for_level(doc_texts, level)
        
        # 记录结果
        for i in range(len(batch["passage_ids"])):
            result = {
                "question_id": batch["question_ids"][i],
                "data_type" :batch["data_types"][i],
                "question": batch["questions"][i],
                "answer": batch["answers"][i],  # 保存answer但不用于生成
                "passage_id": batch["passage_ids"][i],
                "title": batch["titles"][i],
                "segment": batch["segments"][i],
                "is_passage": batch["is_passages"][i],
                f"triples_{level}": triples_list[i]
            }
            
            # 统计空结果和三元组数量
            triples = triples_list[i]
            if not triples:
                empty_count += 1
            triple_counts.append(len(triples))
                
            results.append(result)
    
    # 根据级别设置输出路径
    base_name = os.path.splitext(args.output_path)[0]
    extension = os.path.splitext(args.output_path)[1]
    level_output_path = f"{base_name}_{level}{extension}"
    
    # 保存结果
    kg_generator.save_triples(results, level_output_path)
    
    # 输出详细的统计信息
    total_docs = len(results)
    non_empty = total_docs - empty_count
    avg_triples = sum(triple_counts) / len(triple_counts) if triple_counts else 0
    min_triples = min(triple_counts) if triple_counts else 0
    max_triples = max(triple_counts) if triple_counts else 0
    
    print(f"\n=== {level.upper()} 处理统计 ===")
    print(f"总文档数: {total_docs}")
    print(f"空结果: {empty_count} ({empty_count/total_docs*100:.2f}%)")
    print(f"非空结果: {non_empty} ({non_empty/total_docs*100:.2f}%)")
    print(f"平均三元组数: {avg_triples:.2f}")
    print(f"最少三元组数: {min_triples}")
    print(f"最多三元组数: {max_triples}")
    print(f"结果保存至: {level_output_path}")

def generate_all_kg_triples(args):
    """生成所有级别的三元组（依次运行）"""
    levels = ["few", "medium", "many"]
    
    for level in levels:
        print(f"\n{'='*50}")
        print(f"开始处理 {level.upper()} 级别")
        print(f"{'='*50}")
        generate_kg_triples_for_level(args, level)

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/data/retriever_train_4000_noread_psg_modify10passage.jsonl")
    parser.add_argument('--output_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples/kg_triples.json")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--gpu', type=str, default="7")
    parser.add_argument('--process_passages', type=bool, default=True, help="Whether to process passages in the data")
    parser.add_argument('--level', type=str, choices=['few', 'medium', 'many', 'all'], default='many', 
                       help="Specify which level to generate: few, medium, many, or all")
    args = parser.parse_args()
    
    if args.level == 'all':
        generate_all_kg_triples(args)
    else:
        generate_kg_triples_for_level(args, args.level)
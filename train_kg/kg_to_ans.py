import json
import argparse
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
        
        # 配置vLLM模型
        self.llm = LLM(
            model=model_path,
            dtype='bfloat16',
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
            enable_prefix_caching=False,
            tensor_parallel_size=1,
            enforce_eager=True,
        )
        
        # 所有级别使用相同的生成参数
        self.sampling_params = SamplingParams(
            # temperature=0.3,
            # top_p=0.9,
            # max_tokens=32,  # 所有级别都生成简短答案
            # skip_special_tokens=True,
            # stop=[],
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
            max_tokens= 32,
            logprobs= None,
            prompt_logprobs= None,
            skip_special_tokens= True,
        )
        
        # 所有级别使用相同的指令
        self.instruction = (
            # "Based on the provided knowledge triples, generate a concise and accurate answer to the question. "
            # "Focus on the key information from the triples. "
            # "Keep your answer brief, clear and to the point.\n\n"
            # "Format your answer as a clear, concise statement."
            "Requirements:\n"
            "1. Be concise and to the point\n"
            "2. Answer should be 1-3 sentences maximum\n"
            "3. Use the most relevant information from the source\n"
            "4. Avoid unnecessary details and repetition\n\n"
            "Format: A clear, brief paragraph that directly answers the question."
        )

    def create_answer_prompt(self, question, triples):
        """创建答案生成提示词"""
        triples_text = "\n".join(triples) if triples else "No triples available."
        
        prompt = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": f"Question: {question}\n\nKnowledge Triples:\n{triples_text}\n\nAnswer:"}
        ]
        
        input_prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        return input_prompt

    def generate_answers_for_level(self, data, level, batch_size=20):
        """为指定级别生成答案"""
        print(f"开始为 {level} 级别生成答案...")
        
        all_results = []
        total_items = len(data)
        
        # 分批处理
        for batch_idx in tqdm(range(0, total_items, batch_size), desc=f"处理 {level} 级别"):
            end_idx = min(batch_idx + batch_size, total_items)
            batch_data = data[batch_idx:end_idx]
            
            prompts = []
            item_info = []
            
            for item in batch_data:
                question = item.get("question", "")
                triples = item.get(f"triples_{level}", [])
                
                if not question or not triples:
                    continue
                
                prompt = self.create_answer_prompt(question, triples)
                prompts.append(prompt)
                item_info.append({
                    "question_id": item.get("question_id", ""),
                    "data_type": item.get("data_type", ""),
                    "passage_id": item.get("passage_id", ""),
                    "question": question,
                    "answer": item.get("answer", ""),
                    # "level": level,
                    "triples": triples,  # 保留三元组内容
                    "triple_count": len(triples),
                    "title": item.get("title", ""),
                    "segment": item.get("segment", ""),
                    "is_passage": item.get("is_passage", False)
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
                        **item_info[i],
                        "model_answer": answer
                    })
                
            except Exception as e:
                print(f"批次 {batch_idx//batch_size + 1} 处理失败: {e}")
                continue
        
        return all_results

def generate_answers_from_triples(args):
    """从三元组生成答案的主函数"""
    # 读取并验证三元组数据
    try:
        triples_data = load_and_validate_json(args.triples_path)
    except Exception as e:
        print(f"无法加载文件: {e}")
        return
    
    if not triples_data:
        print("没有找到有效数据")
        return
        
    print(f"成功加载 {len(triples_data)} 条记录")
    
    # 初始化生成器
    answer_generator = AnswerGenerator(
        model_path=args.model_path,
        gpu=args.gpu
    )
    
    # 为指定级别生成答案
    answers = answer_generator.generate_answers_for_level(triples_data, args.level, batch_size=args.batch_size)
    
    # 设置输出路径
    if args.output_path:
        output_path = args.output_path
    else:
        base_name = os.path.splitext(args.triples_path)[0]
        output_path = f"{base_name}_answers_{args.level}.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    print(f"\n=== 生成统计 ===")
    print(f"{args.level} 级别答案数量: {len(answers)}")
    
    # 三元组统计
    total_triples = sum(len(item['triples']) for item in answers)
    avg_triples = total_triples / len(answers) if answers else 0
    print(f"总三元组数量: {total_triples}")
    print(f"平均每个答案的三元组数量: {avg_triples:.2f}")
    
    print(f"答案已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--triples_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/kg_triples/kg_triples_many.json")
    parser.add_argument('--output_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_yuan/kg_to_ans.json", help='输出文件路径，如不指定会自动生成')
    parser.add_argument('--gpu', type=str, default="5")
    parser.add_argument('--batch_size', type=int, default=100, help="批处理大小")
    parser.add_argument('--level', type=str, choices=['few', 'medium', 'many'], default="many")
    args = parser.parse_args()
    
    generate_answers_from_triples(args)
# import json
# import torch
# import os
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForCausalLM, 
#     TrainingArguments, 
#     Trainer,
#     DataCollatorForSeq2Seq
# )
# from peft import LoraConfig, get_peft_model, TaskType
# from datasets import Dataset
# from sklearn.model_selection import train_test_split

# def setup_gpu(gpu_ids):
#     """设置GPU环境"""
#     if gpu_ids:
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
#         print(f"使用GPU: {gpu_ids}")
    
#     # 检查可用GPU数量
#     if torch.cuda.is_available():
#         num_gpus = torch.cuda.device_count()
#         print(f"检测到 {num_gpus} 个GPU")
#         return num_gpus
#     else:
#         print("未检测到GPU，使用CPU")
#         return 0

# def load_training_data(data_path):
#     """加载训练数据"""
#     print(f"正在加载训练数据: {data_path}")
#     with open(data_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     print(f"成功加载 {len(data)} 条训练数据")
#     return data

# def split_train_val(data, val_ratio=0.2):
#     """分割训练集和验证集"""
#     if len(data) < 10:  # 如果数据太少，使用留一法
#         print("数据量较少，使用80%训练，20%验证")
#         train_data, val_data = train_test_split(
#             data, 
#             test_size=val_ratio, 
#             random_state=42,
#             shuffle=True
#         )
#     else:
#         # 尝试按数据类型分层分割
#         data_types = [item.get('data_type', 'unknown') for item in data]
#         try:
#             train_data, val_data = train_test_split(
#                 data, 
#                 test_size=val_ratio, 
#                 random_state=42,
#                 shuffle=True,
#                 stratify=data_types
#             )
#         except:
#             # 如果分层失败，使用普通分割
#             train_data, val_data = train_test_split(
#                 data, 
#                 test_size=val_ratio, 
#                 random_state=42,
#                 shuffle=True
#             )
    
#     print(f"训练集: {len(train_data)} 条")
#     print(f"验证集: {len(val_data)} 条")
    
#     # 显示验证集的数据类型分布
#     val_types = {}
#     for item in val_data:
#         data_type = item.get('data_type', 'unknown')
#         val_types[data_type] = val_types.get(data_type, 0) + 1
#     print(f"验证集数据类型分布: {val_types}")
    
#     return train_data, val_data

# def create_prompt(question, segment, triples):
#     # 限制segment长度以避免内存问题
#     max_segment_length = 1000  # 限制segment长度
#     if len(segment) > max_segment_length:
#         segment = segment[:max_segment_length]
#     """创建训练提示词"""
#     prompt = f"""You are an expert at extracting knowledge triples. Please extract appropriate and key knowledge triples from the given text to help answer the question.

# Question: {question}

# Text:
# {segment}

# Please extract appropriate knowledge triples in the format: <head entity; relation; tail entity>

# Requirements:
# 1. Extract an appropriate number of triples
# 2. Head and tail entities should be meaningful phrases from the text
# 3. Relations should clearly describe the relationship between head and tail entities
# 4. Output format must strictly follow: <head; relation; tail>
# 5. Focus on triples that help understand the core content and answer the question

# Knowledge Triples:
# """
#     return prompt

# def create_completion(triples):
#     """创建完成文本"""
#     completion = "\n".join(triples) + "\n\n"
#     return completion

# def prepare_training_examples(data):
#     """准备训练样本"""
#     training_examples = []
    
#     for item in data:
#         question = item.get('question', '')
#         segment = item.get('segment', '')
#         triples = item.get('triples', [])
        
#         if not question or not segment or not triples:
#             continue
            
#         prompt = create_prompt(question, segment, triples)
#         completion = create_completion(triples)
        
#         training_examples.append({
#             'input': prompt,
#             'output': completion
#         })
    
#     print(f"准备了 {len(training_examples)} 个训练样本")
#     return training_examples

# def tokenize_function(examples, tokenizer, max_length=512):
#     """分词函数"""
#     # 组合输入和输出
#     texts = []
#     for i in range(len(examples['input'])):
#         text = examples['input'][i] + examples['output'][i]
#         texts.append(text)
    
#     # 分词
#     tokenized = tokenizer(
#         texts,
#         truncation=True,
#         padding=False,
#         max_length=max_length,
#         return_tensors=None,
#     )
    
#     # 对于因果语言模型，标签就是输入本身（移位后）
#     tokenized["labels"] = tokenized["input_ids"].copy()
    
#     return tokenized

# def main():
#     import argparse
    
#     # 解析命令行参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', type=str, default="4,5,6,7", help='指定GPU ID，如 "4" 或 "4,5,6,7"')
#     parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct")
#     parser.add_argument('--data_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer/kg_selected_answer.json")
#     parser.add_argument('--output_dir', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/lora_kg_model")
#     parser.add_argument('--batch_size', type=int, default=1, help='单卡批量大小')
#     parser.add_argument('--gradient_accumulation', type=int, default=8, help='梯度累积步数')
#     parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
#     parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的local_rank')
#     args = parser.parse_args()
    
#     # 设置GPU
#     num_gpus = setup_gpu(args.gpu)
    
#     # 计算实际批量大小
#     effective_batch_size = args.batch_size * args.gradient_accumulation * max(1, num_gpus)
#     print(f"实际批量大小: {effective_batch_size} (单卡{args.batch_size} * 累积{args.gradient_accumulation} * GPU数{max(1, num_gpus)})")
    
#     # 加载训练数据
#     raw_data = load_training_data(args.data_path)
    
#     if not raw_data:
#         print("没有可用的训练数据")
#         return
    
#     # 分割训练集和验证集
#     print(f"\n正在分割训练集和验证集 (验证集比例: {args.val_ratio})...")
#     train_data, val_data = split_train_val(raw_data, args.val_ratio)
    
#     # 准备训练和验证样本
#     train_examples = prepare_training_examples(train_data)
#     val_examples = prepare_training_examples(val_data)
    
#     if not train_examples:
#         print("没有可用的训练样本")
#         return
    
#     if not val_examples:
#         print("警告: 没有可用的验证样本，将使用训练集作为验证集")
#         val_examples = train_examples
    
#     # 加载tokenizer和模型
#     print("\n加载tokenizer和模型...")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#     tokenizer.pad_token = tokenizer.eos_token
    
#     # 对于多GPU训练，使用更简单的device_map设置
#     if num_gpus > 1:
#         device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
#         print(f"使用多GPU训练，设备映射: {device_map}")
#     else:
#         device_map = "auto"
#         print("使用单GPU训练，设备映射: auto")
    
#     # 修复模型加载
#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             args.model_path,
#             torch_dtype=torch.bfloat16,
#             device_map=device_map,
#             trust_remote_code=True
#         )
#         print("模型加载成功")
#     except Exception as e:
#         print(f"模型加载失败: {e}")
#         print("尝试不使用device_map加载...")
#         model = AutoModelForCausalLM.from_pretrained(
#             args.model_path,
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True
#         )
#         # 手动移动到GPU
#         if torch.cuda.is_available():
#             model = model.to(f"cuda:{args.gpu.split(',')[0]}")
#             print(f"手动将模型移动到GPU {args.gpu.split(',')[0]}")
    
#     # 配置LoRA
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.05,
#         target_modules=["q_proj", "v_proj"],
#     )
    
#     # 应用LoRA
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
    
#     # 准备训练和验证数据集
#     def create_dataset_dict(examples):
#         return {
#             'input': [ex['input'] for ex in examples],
#             'output': [ex['output'] for ex in examples]
#         }
    
#     train_dataset_dict = create_dataset_dict(train_examples)
#     val_dataset_dict = create_dataset_dict(val_examples)
    
#     train_dataset = Dataset.from_dict(train_dataset_dict)
#     val_dataset = Dataset.from_dict(val_dataset_dict)
    
#     # 分词
#     print("正在对训练集和验证集进行分词...")
#     tokenized_train_dataset = train_dataset.map(
#         lambda examples: tokenize_function(examples, tokenizer, max_length=2048),  # 减小最大长度
#         batched=True,
#         remove_columns=train_dataset.column_names
#     )
    
#     tokenized_val_dataset = val_dataset.map(
#         lambda examples: tokenize_function(examples, tokenizer, max_length=2048),  # 减小最大长度
#         batched=True,
#         remove_columns=val_dataset.column_names
#     )
    
#     # 数据整理器
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer,
#         model=model,
#         padding=True,
#         return_tensors="pt"
#     )
    
#     # 训练参数 - 修复分布式训练参数
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.gradient_accumulation,
#         warmup_steps=100,
#         learning_rate=1e-4,
#         logging_steps=10,
#         eval_steps=500,  # 减少评估频率
#         save_steps=500,  # 减少保存频率
#         eval_strategy="steps",
#         save_strategy="steps",
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         greater_is_better=False,
#         save_total_limit=2,
#         prediction_loss_only=True,
#         remove_unused_columns=False,
#         report_to=None,
#         ddp_find_unused_parameters=False,
        
#         # 分布式训练设置
#         dataloader_pin_memory=False,
#         dataloader_num_workers=0,
#         fp16=False,
#         bf16=True,
        
#         # 分布式训练关键设置
#         local_rank=args.local_rank,
#     )
    
#     # 创建Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train_dataset,
#         eval_dataset=tokenized_val_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )
    
#     # 开始训练
#     print("\n开始训练...")
#     print("训练配置:")
#     print(f"- 训练样本: {len(train_examples)}")
#     print(f"- 验证样本: {len(val_examples)}")
#     print(f"- 评估频率: 每 {training_args.eval_steps} 步")
#     print(f"- 保存频率: 每 {training_args.save_steps} 步")
    
#     train_result = trainer.train()
    
#     # 保存最终模型（只在主进程上保存）
#     if trainer.is_world_process_zero():
#         print("\n保存模型...")
#         trainer.save_model()
#         tokenizer.save_pretrained(args.output_dir)
        
#         # 保存训练指标
#         metrics = train_result.metrics
#         print(f"\n训练完成!")
#         print(f"最终训练损失: {metrics.get('train_loss', 'N/A'):.4f}")
#         print(f"最佳验证损失: {trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else 'N/A'}")
#         print(f"模型保存在: {args.output_dir}")

# if __name__ == "__main__":
#     main()

import json
import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.model_selection import train_test_split

def setup_gpu(gpu_ids):
    """设置GPU环境"""
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"使用GPU: {gpu_ids}")
    
    # 检查可用GPU数量
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU")
        return num_gpus
    else:
        print("未检测到GPU，使用CPU")
        return 0

def load_training_data(data_path):
    """加载训练数据"""
    print(f"正在加载训练数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载 {len(data)} 条训练数据")
    return data

def split_train_val(data, val_ratio=0.2):
    """分割训练集和验证集"""
    if len(data) < 10:  # 如果数据太少，使用留一法
        print("数据量较少，使用80%训练，20%验证")
        train_data, val_data = train_test_split(
            data, 
            test_size=val_ratio, 
            random_state=42,
            shuffle=True
        )
    else:
        # 尝试按数据类型分层分割
        data_types = [item.get('data_type', 'unknown') for item in data]
        try:
            train_data, val_data = train_test_split(
                data, 
                test_size=val_ratio, 
                random_state=42,
                shuffle=True,
                stratify=data_types
            )
        except:
            # 如果分层失败，使用普通分割
            train_data, val_data = train_test_split(
                data, 
                test_size=val_ratio, 
                random_state=42,
                shuffle=True
            )
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    
    # 显示验证集的数据类型分布
    val_types = {}
    for item in val_data:
        data_type = item.get('data_type', 'unknown')
        val_types[data_type] = val_types.get(data_type, 0) + 1
    print(f"验证集数据类型分布: {val_types}")
    
    return train_data, val_data

def create_instruction():
    """创建任务指令"""
    instruction = """
#     You are an expert at extracting knowledge triples. Please extract appropriate and key knowledge triples from the given text to help answer the question.

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

Please extract knowledge triples from the following question and text:
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
"""
    return instruction

def create_input_text(question, segment):
    """创建输入文本"""
    # 限制segment长度以避免内存问题
    max_segment_length = 1000  # 限制segment长度
    if len(segment) > max_segment_length:
        segment = segment[:max_segment_length]
    
    input_text = f"Question: {question}\n\nText:\n{segment}\n\nKnowledge Triples:"
    return input_text

def create_completion(triples):
    """创建完成文本"""
    completion = "\n".join(triples) + "\n\n"
    return completion

def prepare_training_examples(data):
    """准备训练样本"""
    training_examples = []
    instruction = create_instruction()
    
    for item in data:
        question = item.get('question', '')
        segment = item.get('segment', '')
        triples = item.get('triples', [])
        
        if not question or not segment or not triples:
            continue
            
        input_text = create_input_text(question, segment)
        completion = create_completion(triples)
        
        training_examples.append({
            'instruction': instruction,
            'input': input_text,
            'output': completion
        })
    
    print(f"准备了 {len(training_examples)} 个训练样本")
    return training_examples

def tokenize_function(examples, tokenizer, max_length=512):
    """分词函数"""
    # 组合指令、输入和输出
    texts = []
    for i in range(len(examples['input'])):
        text = examples['instruction'][i] + "\n\n" + examples['input'][i] + examples['output'][i]
        texts.append(text)
    
    # 分词
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    
    # 对于因果语言模型，标签就是输入本身（移位后）
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="5,6,7,0", help='指定GPU ID，如 "4" 或 "4,5,6,7"')
    parser.add_argument('--model_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_path', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/data/answer_32/kg_selected_answer.json")
    parser.add_argument('--output_dir', type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/train_kg/lora_kg_model_03")
    parser.add_argument('--batch_size', type=int, default=1, help='单卡批量大小')
    parser.add_argument('--gradient_accumulation', type=int, default=8, help='梯度累积步数')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的local_rank')
    args = parser.parse_args()
    
    # 设置GPU
    num_gpus = setup_gpu(args.gpu)
    
    # 计算实际批量大小
    effective_batch_size = args.batch_size * args.gradient_accumulation * max(1, num_gpus)
    print(f"实际批量大小: {effective_batch_size} (单卡{args.batch_size} * 累积{args.gradient_accumulation} * GPU数{max(1, num_gpus)})")
    
    # 加载训练数据
    raw_data = load_training_data(args.data_path)
    
    if not raw_data:
        print("没有可用的训练数据")
        return
    
    # 分割训练集和验证集
    print(f"\n正在分割训练集和验证集 (验证集比例: {args.val_ratio})...")
    train_data, val_data = split_train_val(raw_data, args.val_ratio)
    
    # 准备训练和验证样本
    train_examples = prepare_training_examples(train_data)
    val_examples = prepare_training_examples(val_data)
    
    if not train_examples:
        print("没有可用的训练样本")
        return
    
    if not val_examples:
        print("警告: 没有可用的验证样本，将使用训练集作为验证集")
        val_examples = train_examples
    
    # 加载tokenizer和模型
    print("\n加载tokenizer和模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 对于多GPU训练，使用更简单的device_map设置
    if num_gpus > 1:
        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        print(f"使用多GPU训练，设备映射: {device_map}")
    else:
        device_map = "auto"
        print("使用单GPU训练，设备映射: auto")
    
    # 修复模型加载
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试不使用device_map加载...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        # 手动移动到GPU
        if torch.cuda.is_available():
            model = model.to(f"cuda:{args.gpu.split(',')[0]}")
            print(f"手动将模型移动到GPU {args.gpu.split(',')[0]}")
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 准备训练和验证数据集
    def create_dataset_dict(examples):
        return {
            'instruction': [ex['instruction'] for ex in examples],
            'input': [ex['input'] for ex in examples],
            'output': [ex['output'] for ex in examples]
        }
    
    train_dataset_dict = create_dataset_dict(train_examples)
    val_dataset_dict = create_dataset_dict(val_examples)
    
    train_dataset = Dataset.from_dict(train_dataset_dict)
    val_dataset = Dataset.from_dict(val_dataset_dict)
    
    # 分词
    print("正在对训练集和验证集进行分词...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length=2048),  # 减小最大长度
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length=2048),  # 减小最大长度
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 训练参数 - 修复分布式训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=100,
        learning_rate=1e-4,
        logging_steps=10,
        eval_steps=1000,  # 减少评估频率
        save_steps=1000,  # 减少保存频率
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        report_to=None,
        ddp_find_unused_parameters=False,
        
        # 分布式训练设置
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        fp16=False,
        bf16=True,
        
        # 分布式训练关键设置
        local_rank=args.local_rank,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("\n开始训练...")
    print("训练配置:")
    print(f"- 训练样本: {len(train_examples)}")
    print(f"- 验证样本: {len(val_examples)}")
    print(f"- 评估频率: 每 {training_args.eval_steps} 步")
    print(f"- 保存频率: 每 {training_args.save_steps} 步")
    
    train_result = trainer.train()
    
    # 保存最终模型（只在主进程上保存）
    if trainer.is_world_process_zero():
        print("\n保存模型...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # 保存训练指标
        metrics = train_result.metrics
        print(f"\n训练完成!")
        print(f"最终训练损失: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"最佳验证损失: {trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else 'N/A'}")
        print(f"模型保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
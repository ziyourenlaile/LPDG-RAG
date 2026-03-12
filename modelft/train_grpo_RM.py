# import torch
# from transformers import (AutoModelForCausalLM, AutoTokenizer,
#                           TrainingArguments as HfTrainingArguments)

# from datasets import Dataset
# from functools import partial
# import logging
# from trl import GRPOTrainer, GRPOConfig
# import transformers
# import json
# from dataclasses import dataclass, field
# from typing import Dict, Optional
# from datasets import load_dataset
# import re
# from rouge import Rouge
# import random
# from template import (
#     IGNORE_INDEX,
#     PROMPT_DICT,
#     user_tokens,
#     assistant_tokens,
#     pythia_user_tokens,
#     pythia_assistant_tokens, RESPONSE_START_TOKEN_IDS,
# )
# logger = logging.getLogger(__name__)

# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default="Meta-Llama-3-8B-Instruct")
#     llama_style: bool = field(default=True)
#     use_template: bool = field(default=True)


# @dataclass
# class DataArguments:
#     train_data_path: str = field(
#         default=None,
#         metadata={"help": "Path to the training data."},
#     )
#     eval_data_path: str = field(
#         default=None,
#         metadata={"help": "Path to the test data."},
#     )
#     max_prompt_lengths: int = field(default=1500, metadata={"help": "Maximum prompt sequence length."}, )
#     top_n: int = field(default=10, metadata={"help": "how many psg use."}, )


# @dataclass
# class TrainingArguments(GRPOConfig):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     load_lora_model: bool = field(default=True)

#     ref_model: Optional[str] = field(default=None)
#     use_lora: bool = field(default=True)
#     output_dir: str = field(default=None)
#     save_steps: int = field(default=50)
#     eval_steps: int = field(default=50)
#     per_device_train_batch_size: int = field(default=2)
#     per_device_eval_batch_size: int = field(default=2)
#     evaluation_strategy: str = field(default='steps')
#     logging_steps: int = field(default=3)
#     logging_dir: str = field(default=None)
#     bf16: bool = field(default=True)
#     beta: float = field(default=0.1, metadata={"help": "DPO beta parameter (保留兼容，GRPO暂不用)"})
#     gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
#     warmup_steps: int = field(default=100, metadata={"help": "Warmup steps"})
#     lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type"})
#     gradient_checkpointing: bool = field(default=True, metadata={"help": "Use gradient checkpointing"})
#     num_generations: int = field(
#         default=2,
#         metadata={"help": "Number of generations per sample in GRPO, must divide generation_batch_size"}
#     )
#     max_length: int = field(default=3048, metadata={"help": "Maximum sequence length for model input during training"})
#     max_prompt_length: int = field(default=2500, metadata={"help": "Maximum prompt length for model input during training"})


# def load_model_and_tokenizer(
#         model_path: str,
#         llama_style: bool,
#         use_lora: bool = True,
#         bf16: bool = False,
#         fp16: bool = False,
#         load_lora_model: bool = False,
# ):
#     """load model and tokenizer"""
#     assert not (bf16 and fp16), "bf16 or fp16, not both"
#     if bf16:
#         dtype = torch.bfloat16
#     elif fp16:
#         dtype = torch.float16
#     else:
#         dtype = torch.float32

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=dtype,
#         trust_remote_code=True,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

#     tokenizer.pad_token = tokenizer.eos_token
#     if use_lora:
#         from peft import LoraConfig, TaskType, get_peft_model

#         if llama_style:
#             lora_config = LoraConfig(
#                 task_type=TaskType.CAUSAL_LM,
#                 r=8,
#                 lora_alpha=32,
#                 lora_dropout=0.1,
#                 inference_mode=False,
#             )
#         else:
#             lora_config = LoraConfig(
#                 init_lora_weights="gaussian",
#                 task_type=TaskType.CAUSAL_LM,
#                 target_modules=["q_proj", "v_proj"],
#                 r=8,
#                 lora_alpha=32,
#                 lora_dropout=0.1,
#                 inference_mode=False,
#             )
#         model = get_peft_model(model, lora_config)
#         model.print_trainable_parameters()
#         model.enable_input_require_grads()

#     return model, tokenizer


# def _rougel_score(prediction, ground_truth):
#     """计算ROUGE-L分数"""
#     rouge = Rouge()
#     try:
#         scores = rouge.get_scores(prediction, ground_truth, avg=True)
#     except ValueError:  # "Hypothesis is empty."
#         return 0.0
#     return scores["rouge-l"]["f"]


# def is_answer_correct(data_type, model_answer, ground_truth):
#     """
#     判断模型答案是否正确
#     """
#     # 处理可能的None值
#     if model_answer is None or ground_truth is None:
#         return False

#     model_answer = str(model_answer).lower().strip()
#     ground_truth = str(ground_truth).lower().strip()

#     # 第一类数据类型：精确匹配
#     if data_type in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa', 'gsm8k', 'strategyqa', 'web_questions']:
#         match_result = ground_truth in model_answer
#         return match_result
#     # 第二类数据类型：ROUGE-L评分
#     elif data_type in ['wiki_qa', 'yahoo_answers_qa', 'marcoqa']:
#         score = _rougel_score(model_answer, ground_truth)
#         return score > 0.22
#     else:
#         # 默认使用精确匹配
#         return ground_truth in model_answer


# def load_reward_model(model_path, base_model_path="Meta-Llama-3-8B-Instruct"):
#     """
#     加载奖励模型（基础模型 + LoRA权重）
#     """
#     from peft import PeftModel, PeftConfig
    
#     # 加载基础模型
#     reward_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
#     reward_model = AutoModelForCausalLM.from_pretrained(
#         base_model_path,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#     )
    
#     # 加载LoRA配置和权重
#     reward_model = PeftModel.from_pretrained(
#         reward_model,
#         model_path,
#         torch_dtype=torch.bfloat16
#     )
    
#     # 添加pad token
#     if reward_tokenizer.pad_token is None:
#         reward_tokenizer.pad_token = reward_tokenizer.eos_token

#     return reward_model, reward_tokenizer


# def prepare_rm_input(query, response_a, response_b,current_ground_truth):
#     """
#     准备奖励模型的输入格式
#     """
#     prompt1 = f'Question: {query}\n\nResponse A: {response_a}\n\nResponse B: {response_b}\n\nIs Response A better than Response B? ground_truth: {current_ground_truth}\n Answer with only "yes" or "no":\n'
#     prompt2 = f'Question: {query}\n\nResponse A: {response_b}\n\nResponse B: {response_a}\n\nIs Response A better than Response B? ground_truth: {current_ground_truth}\n Answer with only "yes" or "no":\n'
#     return prompt1, prompt2


# def calculate_reward_with_rm(reward_model, reward_tokenizer, query, generated_response, reference_response,current_ground_truth):
#     """
#     使用奖励模型计算奖励值
#     """
#     # 准备两个对比输入
#     prompt1, prompt2 = prepare_rm_input(query, generated_response, reference_response,current_ground_truth)

#     # Tokenize prompts
#     inputs1 = reward_tokenizer(
#         prompt1,
#         truncation=True,
#         padding=False,
#         max_length=500,
#         add_special_tokens=True
#     )
#     inputs2 = reward_tokenizer(
#         prompt2,
#         truncation=True,
#         padding=False,
#         max_length=500,
#         add_special_tokens=True
#     )

#     # Get token IDs for "yes" and "no"
#     yes_token_id = reward_tokenizer.convert_tokens_to_ids("yes")
#     no_token_id = reward_tokenizer.convert_tokens_to_ids("no")

#     # Process first prompt
#     prompt1_ids = inputs1["input_ids"]
#     prompt2_ids = inputs2["input_ids"]

#     # Add label tokens
#     full_input1_ids = prompt1_ids + [yes_token_id]
#     full_input2_ids = prompt2_ids + [no_token_id]

#     attention_mask1 = [1] * len(full_input1_ids)
#     attention_mask2 = [1] * len(full_input2_ids)

#     # Pad to same length
#     max_len = max(len(full_input1_ids), len(full_input2_ids))
#     pad_token_id = reward_tokenizer.pad_token_id

#     padded_input1 = full_input1_ids + [pad_token_id] * (max_len - len(full_input1_ids))
#     padded_input2 = full_input2_ids + [pad_token_id] * (max_len - len(full_input2_ids))
#     padded_mask1 = attention_mask1 + [0] * (max_len - len(attention_mask1))
#     padded_mask2 = attention_mask2 + [0] * (max_len - len(attention_mask2))

#     # Create batch tensors
#     input_ids_batch = torch.tensor([padded_input1, padded_input2], dtype=torch.long).to(reward_model.device)
#     attention_mask_batch = torch.tensor([padded_mask1, padded_mask2], dtype=torch.long).to(reward_model.device)

#     with torch.no_grad():
#         outputs = reward_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
#         logits = outputs.logits  # [2, seq_len, vocab_size]

#         # Get logits for the last token (the "yes"/"no" prediction)
#         label_pos1 = len(padded_mask1) - 1
#         label_pos2 = len(padded_mask2) - 1

#         # Predictions for each prompt
#         pred_logits1 = logits[0, label_pos1 - 1]  # Logits for predicting the "yes" token
#         pred_logits2 = logits[1, label_pos2 - 1]  # Logits for predicting the "no" token

#         # Calculate probabilities
#         prob1 = torch.softmax(pred_logits1, dim=-1)[yes_token_id]  # P(yes | A > B)
#         prob2 = torch.softmax(pred_logits2, dim=-1)[no_token_id]   # P(no | B > A)

#         # Combine the probabilities as a reward signal
#         # Higher probability that A > B and B < A indicates A is better
#         combined_prob = (prob1 + prob2) / 2.0
#         reward = combined_prob.item()

#     return reward


# def preference_reward_func_with_rm(
#         completions: list,
#         # prompt: list,
#         data_type: list,
#         ground_truth: list,
#         model_self_correct: list,
#         model_self_answer: list,
#         correct_passages: list,
#         valid_cots: list,
#         reward_model=None,
#         reward_tokenizer=None,
#         queries=None,  # 新增queries参数
#         reference_responses=None,  # 新增reference_responses参数
#         **kwargs
# ) -> list:
#     """
#     使用奖励模型的奖励函数
#     """
#     rewards = []

#     for i in range(len(completions)):
#         generated_response = completions[i]
#         query = queries[i] if queries and i < len(queries) else ""
#         reference_response = reference_responses[i] if reference_responses and i < len(reference_responses) else ""
#         current_ground_truth = ground_truth[i] if i < len(ground_truth) else ''
        
#         if reward_model is not None and reward_tokenizer is not None:
#             # 使用奖励模型计算奖励
#             reward = calculate_reward_with_rm(
#                 reward_model,
#                 reward_tokenizer,
#                 query,
#                 generated_response,
#                 reference_response,
#                 current_ground_truth
#             )
#         else:
#             # 备用奖励计算方法
#             print("111111111111111111111111111111111111111111111")
#             current_data_type = data_type[i] if i < len(data_type) else 'unknown'
#             current_ground_truth = ground_truth[i] if i < len(ground_truth) else ''
#             answer_correct = is_answer_correct(current_data_type, generated_response, current_ground_truth)
#             reward = 1.0 if answer_correct else 0.0

#         rewards.append(reward)

#     return rewards


# def preprocessing(example, args, tokenizer):
#     """改进的数据预处理函数"""
#     one_item = {}
#     datatype = example['data_type']

#     # 选择合适的模板
#     if datatype in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa']:
#         template = PROMPT_DICT['Mutichoice_querypassage_to_CoT']
#     else:
#         template = PROMPT_DICT['QA_querypassage_to_CoT']

#     query = example['query']
#     psgs = example['passages'][:args.top_n]
#     psg_list = []

#     # 构建 passages 文本，并收集有效的参考回答
#     valid_cots = []  # 存储有效的COT用于后续比较
#     reference_responses = []  # 存储参考回答

#     for p in psgs:
#         p_text = p.get('segment', '')
#         p_id = p.get('id', '')
#         p_cot = p.get('model_answer', '')  # 参考回答来自这里
#         p_is_correct = p.get('is_correct', False)

#         if isinstance(p_text, str):
#             psg_list.append(f"{p_text}\n")
#             # 收集有效的参考回答
#             if p_cot and len(p_cot.strip()) > 0:
#                 reference_responses.append(p_cot)
#             # 提取有效的COT（is_correct为True的COT）
#             if p_is_correct and p_cot and len(p_cot.strip()) > 0:
#                 valid_cots.append({
#                     'id': p_id,
#                     'cot': p_cot,
#                     'is_correct': p_is_correct
#                 })

#     aug_psg = '\n'.join(psg_list)
#     token_query = tokenizer([query])
#     query_length = len(token_query.input_ids[0])
#     token_aug_psg = tokenizer([aug_psg])
#     token_aug_psg_truncated = token_aug_psg.input_ids[0][:args.max_prompt_lengths - 32 - query_length]
#     new_aug_psg = tokenizer.decode(token_aug_psg_truncated, skip_special_tokens=True)

#     # 构建输入格式
#     input_data = template.format(passages=new_aug_psg, question=query)
#     aug_query = [{"role": "user", "content": input_data}]
#     aug_query = tokenizer.apply_chat_template(
#         aug_query,
#         add_generation_prompt=True,
#         tokenize=False
#     )

#     one_item["prompt"] = aug_query
#     one_item["query"] = query  # 保存原始query用于奖励模型
#     one_item["data_type"] = datatype
#     one_item["ground_truth"] = example["ground_truth"]
#     one_item["model_self_correct"] = example["model_self_correct"]
#     one_item["model_self_answer"] = example["model_self_answer"]
#     one_item["correct_passages"] = example.get("correct_passages", [])
#     one_item["valid_cots"] = valid_cots  # 存储有效的COT用于奖励计算

#     # 随机选择一个参考回答
#     if reference_responses:
#         selected_reference = random.choice(reference_responses)
#     else:
#         selected_reference = ""  # 如果没有参考回答，则为空字符串
#     one_item["reference_response"] = selected_reference

#     return one_item

# def create_reward_func_with_rm(reward_model, reward_tokenizer):
#     """
#     创建包含奖励模型的奖励函数
#     """
#     def reward_func(completions, data_type, ground_truth, model_self_correct, 
#                    model_self_answer, correct_passages, valid_cots, queries=None, 
#                    reference_responses=None, **kwargs):
#         return preference_reward_func_with_rm(
#             completions=completions,
#             # prompt=prompt,
#             data_type=data_type,
#             ground_truth=ground_truth,
#             model_self_correct=model_self_correct,
#             model_self_answer=model_self_answer,
#             correct_passages=correct_passages,
#             valid_cots=valid_cots,
#             reward_model=reward_model,
#             reward_tokenizer=reward_tokenizer,
#             queries=queries,
#             reference_responses=reference_responses
#         )
#     return reward_func



# def extract_queries_from_dataset(dataset):
#     """从数据集中提取所有query"""
#     queries = []
#     for item in dataset:
#         queries.append(item["query"])
#     return queries


# def extract_reference_responses_from_dataset(dataset):
#     """从数据集中提取所有参考回答"""
#     reference_responses = []
#     for item in dataset:
#         reference_responses.append(item["reference_response"])
#     return reference_responses


# if __name__ == "__main__":
#     print("Starting GRPO training with reward model...")
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments)
#     )

#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     print("Model Arguments:", model_args)

#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO
#     )
#     logger.info("Training/evaluation parameters %s", training_args)
#     logger.info("MODEL parameters %s", model_args)
#     logger.info("DATA parameters %s", data_args)

#     # 加载主模型和分词器
#     model, tokenizer = load_model_and_tokenizer(
#         model_path=model_args.model_name_or_path,
#         llama_style=model_args.llama_style,
#         use_lora=training_args.use_lora,
#         bf16=training_args.bf16,
#         fp16=getattr(training_args, "fp16", False),
#         load_lora_model=training_args.load_lora_model
#     )

#     # 加载奖励模型（使用训练好的RM路径）
#     print("Loading reward model...")
#     reward_model, reward_tokenizer = load_reward_model(
#         "/srv/nfs/home/njnu_zrq/RankCoT/src/modelft/RM_train/pairwise_rm_lora_fixed/epoch_2",  # 替换为实际的RM路径
#         "/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct"  # 基础模型路径
#     )

#     # 数据预处理
#     partial_preprocess = partial(preprocessing, args=data_args, tokenizer=tokenizer)

#     # 加载并预处理训练集
#     train_dataset = load_dataset("json", data_files=data_args.train_data_path, split="train")
#     train_dataset = train_dataset.map(partial_preprocess, remove_columns=train_dataset.column_names)

#     # 加载并预处理验证集
#     eval_dataset = load_dataset("json", data_files=data_args.eval_data_path, split="train")
#     eval_dataset = eval_dataset.map(partial_preprocess, remove_columns=eval_dataset.column_names)

#     # 查看训练集数据
#     print("===== 训练集前2条数据 =====")
#     for i in range(min(2, len(train_dataset))):
#         sample = train_dataset[i]
#         print(f"第{i + 1}条数据:")
#         print(f"prompt: {sample['prompt'][:250]}..." if len(sample['prompt']) > 250 else f"prompt: {sample['prompt']}")
#         print(f"query: {sample['query']}")
#         print(f"reference_response: {sample['reference_response'][:100]}..." if len(sample['reference_response']) > 100 else f"reference_response: {sample['reference_response']}")
#         print(f"data_type: {sample['data_type']}")
#         print(f"ground_truth: {sample['ground_truth']}")
#         print(f"model_self_correct: {sample['model_self_correct']}")
#         print("-" * 80)

#     # 从数据集中提取queries和reference_responses
#     train_queries = extract_queries_from_dataset(train_dataset)
#     train_reference_responses = extract_reference_responses_from_dataset(train_dataset)
#     eval_queries = extract_queries_from_dataset(eval_dataset)
#     eval_reference_responses = extract_reference_responses_from_dataset(eval_dataset)

#     # 创建包含奖励模型的奖励函数
#     reward_func_with_rm = create_reward_func_with_rm(reward_model, reward_tokenizer)

#     # GRPO训练器初始化
#     grpo_trainer = GRPOTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         processing_class=tokenizer,
#         reward_funcs=reward_func_with_rm,
#     )

#     # 开始训练
#     logger.info("Starting GRPO training with reward model...")
#     grpo_trainer.train()

#     # 保存模型
#     logger.info("Saving trained model...")
#     grpo_trainer.save_model(training_args.output_dir)
#     logger.info(f"Model saved to {training_args.output_dir}")


import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments as HfTrainingArguments)
from datasets import Dataset
from functools import partial
import logging
from trl import GRPOTrainer, GRPOConfig
import transformers
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datasets import load_dataset
import re
from rouge import Rouge
import random
from template import (
    IGNORE_INDEX,
    PROMPT_DICT,
    user_tokens,
    assistant_tokens,
    pythia_user_tokens,
    pythia_assistant_tokens, RESPONSE_START_TOKEN_IDS,
)

logger = logging.getLogger(__name__)

# ========== 原有类定义保持不变 ==========
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Meta-Llama-3-8B-Instruct")
    llama_style: bool = field(default=True)
    use_template: bool = field(default=True)

@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the test data."},
    )
    max_prompt_lengths: int = field(default=1500, metadata={"help": "Maximum prompt sequence length."}, )
    top_n: int = field(default=10, metadata={"help": "how many psg use."}, )

@dataclass
class TrainingArguments(GRPOConfig):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_lora_model: bool = field(default=True)
    ref_model: Optional[str] = field(default=None)
    use_lora: bool = field(default=True)
    output_dir: str = field(default=None)
    save_steps: int = field(default=50)
    eval_steps: int = field(default=50)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    evaluation_strategy: str = field(default='steps')
    logging_steps: int = field(default=3)
    logging_dir: str = field(default=None)
    bf16: bool = field(default=True)
    beta: float = field(default=0.1, metadata={"help": "DPO beta parameter (保留兼容，GRPO暂不用)"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    warmup_steps: int = field(default=100, metadata={"help": "Warmup steps"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Use gradient checkpointing"})
    num_generations: int = field(
        default=2,
        metadata={"help": "Number of generations per sample in GRPO, must divide generation_batch_size"}
    )
    max_length: int = field(default=3048, metadata={"help": "Maximum sequence length for model input during training"})
    max_prompt_length: int = field(default=2500, metadata={"help": "Maximum prompt length for model input during training"})


# ========== 原有辅助函数保持不变 ==========
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:
        return 0.0
    return scores["rouge-l"]["f"]

def is_answer_correct(data_type, model_answer, ground_truth):
    if model_answer is None or ground_truth is None:
        return False
    model_answer = str(model_answer).lower().strip()
    ground_truth = str(ground_truth).lower().strip()
    if data_type in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa', 'gsm8k', 'strategyqa', 'web_questions']:
        return ground_truth in model_answer
    elif data_type in ['wiki_qa', 'yahoo_answers_qa', 'marcoqa']:
        score = _rougel_score(model_answer, ground_truth)
        return score > 0.22
    else:
        return ground_truth in model_answer

def load_model_and_tokenizer(
    model_path: str,
    llama_style: bool,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
    load_lora_model: bool = False,
):
    assert not (bf16 and fp16), "bf16 or fp16, not both"
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        if llama_style:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                inference_mode=False,
            )
        else:
            lora_config = LoraConfig(
                init_lora_weights="gaussian",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"],
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                inference_mode=False,
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.enable_input_require_grads()

    return model, tokenizer


# ========== 新增：批量 RM 推理函数 ==========
def prepare_rm_inputs_batch(queries, responses_a, responses_b, ground_truths):
    criteria = (
            # "When evaluating, consider the following criteria:\n"
            # "1. Does the response contain the correct answer?\n"
            # "2. How similar is the response content to the ground truth?\n"
            # "3. Is the response shorter and more concise?\n"
            # "4. Does the response include a clear reasoning or thought process?\n"
            # "5. Is the correct answer presented at the end of the response?\n"
            """
            When evaluating a response, consider the following criteria:
            Accuracy: Does the response contain the correct answer?
            Fidelity: How closely does the response align with the ground truth in terms of content and meaning?
            Conciseness: Is the response clear and succinct, avoiding unnecessary elaboration?
            Reasoning: Does the response include a logical and transparent thought process or justification?
            Clarity of Conclusion: Is the correct answer clearly stated—preferably at the end of the response?
            Recoverability: Can the ground truth be reliably inferred or reconstructed solely from the response?
            """
        )    
    
    prompts1, prompts2 = [], []
    for q, a, b, gt in zip(queries, responses_a, responses_b, ground_truths):
        p1 = (
            f'Question: {q}\n\n'
            f'Response A: {a}\n\n'
            f'Response B: {b}\n\n'
            f'Ground truth: {gt}\n\n'
            f'{criteria}'
            f'Is Response A better than Response B according to the above criteria?\n\n'
            f'Answer with only "yes" or "no".\n'
        )
        # p1 = f'Question: {q}\n\nResponse A: {a}\n\nResponse B: {b}\n\nIs Response A better than Response B? ground_truth: {gt}\nAnswer with only "y" or "n":\n'
        p2 = (
            f'Question: {q}\n\n'
            f'Response A: {b}\n\n'
            f'Response B: {a}\n\n'
            f'Ground truth: {gt}\n\n'
            f'{criteria}'
            f'Is Response A better than Response B according to the above criteria?\n\n'
            f'Answer with only "yes" or "no".\n'
        )
        # p2 = f'Question: {q}\n\nResponse A: {b}\n\nResponse B: {a}\n\nIs Response A better than Response B? ground_truth: {gt}\nAnswer with only "y" or "n":\n'
        prompts1.append(p1)
        prompts2.append(p2)
    return prompts1, prompts2

def calculate_rewards_with_rm_batch(
    reward_model,
    reward_tokenizer,
    queries: List[str],
    generated_responses: List[str],
    reference_responses: List[str],
    ground_truths: List[str],
    max_length: int = 500,
):
    device = reward_model.device
    N = len(queries)
    if N == 0:
        return []

    prompts1, prompts2 = prepare_rm_inputs_batch(queries, generated_responses, reference_responses, ground_truths)

    # Tokenize
    inputs1 = reward_tokenizer(
        prompts1,
        truncation=True,
        padding=True,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors="pt"
    ).to(device)

    inputs2 = reward_tokenizer(
        prompts2,
        truncation=True,
        padding=True,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors="pt"
    ).to(device)

    # Use "y" and "n" tokens (Llama supports these as single tokens)
    y_token_id = reward_tokenizer.convert_tokens_to_ids("yes")
    n_token_id = reward_tokenizer.convert_tokens_to_ids("no")
    if y_token_id == reward_tokenizer.unk_token_id or n_token_id == reward_tokenizer.unk_token_id:
        raise RuntimeError("Tokenizer missing 'yes' or 'no' tokens!")

    with torch.no_grad(), torch.inference_mode():
        outputs1 = reward_model(**inputs1)
        outputs2 = reward_model(**inputs2)

        logits1 = outputs1.logits
        logits2 = outputs2.logits

        # Get last non-pad position
        seq_len1 = inputs1["attention_mask"].sum(dim=1)  # [N]
        seq_len2 = inputs2["attention_mask"].sum(dim=1)  # [N]

        batch_idx = torch.arange(N, device=device)
        pred_logits1 = logits1[batch_idx, seq_len1 - 1]  # [N, vocab]
        pred_logits2 = logits2[batch_idx, seq_len2 - 1]  # [N, vocab]

        prob_y_A_better = torch.softmax(pred_logits1, dim=-1)[:, y_token_id]   # P(y | A > B)
        prob_n_B_better = torch.softmax(pred_logits2, dim=-1)[:, n_token_id]   # P(n | B > A) → i.e., P(A > B)

        combined = (prob_y_A_better + prob_n_B_better) / 2.0
        # combined = prob_y_A_better
        return combined.cpu().float().tolist()


def preference_reward_func_with_rm_batch(
    completions: list,
    data_type: list,
    ground_truth: list,
    model_self_correct: list,
    model_self_answer: list,
    correct_passages: list,
    valid_cots: list,
    reward_model=None,
    reward_tokenizer=None,
    queries=None,
    reference_responses=None,
    **kwargs
) -> list:
    if not completions:
        return []

    # Assume all completions belong to the same query (GRPO behavior)
    query = queries[0] if queries else ""
    ref_resp = reference_responses[0] if reference_responses else ""
    gt = ground_truth[0] if ground_truth else ""

    batch_queries = [query] * len(completions)
    batch_refs = [ref_resp] * len(completions)
    batch_gts = [gt] * len(completions)

    if reward_model is not None and reward_tokenizer is not None:
        try:
            rewards = calculate_rewards_with_rm_batch(
                reward_model,
                reward_tokenizer,
                batch_queries,
                completions,
                batch_refs,
                batch_gts,
                max_length=500
            )
        except Exception as e:
            logger.warning(f"[RM Error] Falling back to rule-based reward: {e}")
            rewards = []
            dt = data_type[0] if data_type else 'unknown'
            for comp in completions:
                correct = is_answer_correct(dt, comp, gt)
                rewards.append(1.0 if correct else 0.0)
    else:
        # Fallback
        rewards = []
        dt = data_type[0] if data_type else 'unknown'
        for comp in completions:
            correct = is_answer_correct(dt, comp, gt)
            rewards.append(1.0 if correct else 0.0)

    return rewards

def create_reward_func_with_rm(reward_model, reward_tokenizer):
    def reward_func(**kwargs):
        return preference_reward_func_with_rm_batch(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            **kwargs
        )
    return reward_func


# ========== 数据预处理等保持不变 ==========
def preprocessing(example, args, tokenizer):
    one_item = {}
    datatype = example['data_type']

    if datatype in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa']:
        template = PROMPT_DICT['Mutichoice_querypassage_to_CoT']
    else:
        template = PROMPT_DICT['QA_querypassage_to_CoT']

    query = example['query']
    psgs = example['passages'][:args.top_n]
    psg_list = []
    valid_cots = []
    reference_responses = []

    for p in psgs:
        p_text = p.get('segment', '')
        p_id = p.get('id', '')
        p_cot = p.get('model_answer', '')
        p_is_correct = p.get('is_correct', False)

        if isinstance(p_text, str):
            psg_list.append(f"{p_text}\n")
            if p_cot and len(p_cot.strip()) > 0:
                reference_responses.append(p_cot)
            if p_is_correct and p_cot and len(p_cot.strip()) > 0:
                valid_cots.append({
                    'id': p_id,
                    'cot': p_cot,
                    'is_correct': p_is_correct
                })

    aug_psg = '\n'.join(psg_list)
    token_query = tokenizer([query])
    query_length = len(token_query.input_ids[0])
    token_aug_psg = tokenizer([aug_psg])
    token_aug_psg_truncated = token_aug_psg.input_ids[0][:args.max_prompt_lengths - 32 - query_length]
    new_aug_psg = tokenizer.decode(token_aug_psg_truncated, skip_special_tokens=True)

    input_data = template.format(passages=new_aug_psg, question=query)
    aug_query = [{"role": "user", "content": input_data}]
    aug_query = tokenizer.apply_chat_template(
        aug_query,
        add_generation_prompt=True,
        tokenize=False
    )

    one_item["prompt"] = aug_query
    one_item["query"] = query
    one_item["data_type"] = datatype
    one_item["ground_truth"] = example["ground_truth"]
    one_item["model_self_correct"] = example["model_self_correct"]
    one_item["model_self_answer"] = example["model_self_answer"]
    one_item["correct_passages"] = example.get("correct_passages", [])
    one_item["valid_cots"] = valid_cots
    if(example["model_self_correct"] == "true"):
        reference_responses.append(example["model_self_answer"])
    if reference_responses:
        selected_reference = random.choice(reference_responses)
    else:
        selected_reference = ""
    one_item["reference_response"] = selected_reference

    return one_item

def extract_queries_from_dataset(dataset):
    return [item["query"] for item in dataset]

def extract_reference_responses_from_dataset(dataset):
    return [item["reference_response"] for item in dataset]

def load_reward_model(model_path, base_model_path="Meta-Llama-3-8B-Instruct"):
    from peft import PeftModel
    reward_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    reward_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    reward_model = PeftModel.from_pretrained(
        reward_model,
        model_path,
        torch_dtype=torch.bfloat16
    )
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    return reward_model, reward_tokenizer


# ========== 主函数保持结构，仅微调 ==========
if __name__ == "__main__":
    print("Starting GRPO training with batched reward model...")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Load main model
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        llama_style=model_args.llama_style,
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=getattr(training_args, "fp16", False),
        load_lora_model=training_args.load_lora_model
    )

    # Load reward model
    print("Loading reward model...")
    reward_model, reward_tokenizer = load_reward_model(
        "/srv/nfs/home/njnu_zrq/RankCoT/src/modelft/RM_train/llama3_8B/epoch_2",
        "/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct"
    )
    reward_model.eval()  # Important!
    reward_model.to(model.device)  # Put on same device as main model

    # Data preprocessing
    partial_preprocess = partial(preprocessing, args=data_args, tokenizer=tokenizer)

    train_dataset = load_dataset("json", data_files=data_args.train_data_path, split="train")
    train_dataset = train_dataset.map(partial_preprocess, remove_columns=train_dataset.column_names)

    eval_dataset = load_dataset("json", data_files=data_args.eval_data_path, split="train")
    eval_dataset = eval_dataset.map(partial_preprocess, remove_columns=eval_dataset.column_names)

    # Create reward function
    reward_func_with_rm = create_reward_func_with_rm(reward_model, reward_tokenizer)

    # Initialize trainer
    grpo_trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func_with_rm,
    )

    # Train
    logger.info("Starting GRPO training with batched reward model...")
    grpo_trainer.train()

    # Save
    logger.info("Saving trained model...")
    grpo_trainer.save_model(training_args.output_loradir)
    logger.info(f"Model saved to {training_args.output_dir}")

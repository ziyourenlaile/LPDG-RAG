import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)
from accelerate import Accelerator
from tqdm import tqdm
import argparse
from peft import LoraConfig, get_peft_model, TaskType

class PairwiseRMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                query = item["query"]
                chosen = item["model_answer"]["chosen"]
                rejected = item["model_answer"]["rejected"]
                ground_truth = item["ground_truth"]
                self.examples.append((query, chosen, rejected ,ground_truth))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, chosen, rejected, ground_truth = self.examples[idx]
        # "When evaluating, consider the following criteria:\n"
            # "1. Does the response contain the correct answer?\n"
            # "2. How similar is the response content to the ground truth?\n"
            # "3. Is the response shorter and more concise?\n"
            # "4. Does the response include a clear reasoning or thought process?\n"
            # "5. Is the correct answer presented at the end of the response?\n"
        criteria = (
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

        prompt1 = (
            f'Question: {query}\n\n'
            f'Response A: {chosen}\n\n'
            f'Response B: {rejected}\n\n'
            f'Ground truth: {ground_truth}\n\n'
            f'{criteria}'
            f'Is Response A better than Response B according to the above criteria?\n\n'
            f'Answer with only "yes" or "no".\n'
        )
        
        prompt2 = (
            f'Question: {query}\n\n'
            f'Response A: {rejected}\n\n'
            f'Response B: {chosen}\n\n'
            f'Ground truth: {ground_truth}\n\n'
            f'{criteria}'
            f'Is Response A better than Response B according to the above criteria?\n\n'
            f'Answer with only "yes" or "no".\n'
        )
        
        return [
            {"prompt": prompt1, "label": "yes"},
            {"prompt": prompt2, "label": "no"}
        ]
    # def __getitem__(self, idx):
    #     query, chosen, rejected, ground_truth = self.examples[idx]
    #     # Using the same prompt format as provided
    #     prompt1 = f'Question: {query}\n\nResponse A: {chosen}\n\nResponse B: {rejected}\n\nIs Response A better than Response B?\n\n ground_truth: {ground_truth}\n\n Answer with only "yes" or "no".\n'
        
        
        
    #     prompt2 = f'Question: {query}\n\nResponse A: {rejected}\n\nResponse B: {chosen}\n\nIs Response A better than Response B?\n\n ground_truth: {ground_truth}\n\n Answer with only "yes" or "no".\n'
    #     return [
    #         {"prompt": prompt1, "label": "yes"},
    #         {"prompt": prompt2, "label": "no"}
    #     ]

def collate_fn(batch, tokenizer, max_length):
    flat_examples = []
    for pair in batch:
        flat_examples.extend(pair)

    prompts = [ex["prompt"] for ex in flat_examples]
    labels = [ex["label"] for ex in flat_examples]

    # Tokenize prompts
    prompt_encodings = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=max_length - 2,
        add_special_tokens=True 
    )

    input_ids_list = []
    attention_mask_list = []
    label_ids_list = []

    yes_token_id = tokenizer.convert_tokens_to_ids("yes")
    no_token_id = tokenizer.convert_tokens_to_ids("no")
    pad_token_id = tokenizer.pad_token_id

    for i, label in enumerate(labels):
        prompt_ids = prompt_encodings["input_ids"][i]
        label_id = yes_token_id if label == "yes" else no_token_id

        # Append label token
        full_input_ids = prompt_ids + [label_id]
        attention_mask = [1] * len(full_input_ids)

        # Truncate if necessary
        if len(full_input_ids) > max_length:
            full_input_ids = full_input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]

        input_ids_list.append(full_input_ids)
        attention_mask_list.append(attention_mask)
        label_ids_list.append(label_id)

    # Dynamic padding (Right Padding is fine for training as long as we index correctly)
    max_len = max(len(ids) for ids in input_ids_list)
    input_ids_padded = [ids + [pad_token_id] * (max_len - len(ids)) for ids in input_ids_list]
    attention_mask_padded = [mask + [0] * (max_len - len(mask)) for mask in attention_mask_list]

    input_ids = torch.tensor(input_ids_padded, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_padded, dtype=torch.long)
    labels_tensor = torch.tensor(label_ids_list, dtype=torch.long)

    # Calculate the position of the label token
    # sum(mask) gives length. Index is length - 1.
    seq_lengths = attention_mask.sum(dim=1)
    label_indices = seq_lengths - 1 

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_indices": label_indices,
        "labels": labels_tensor
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/modelft/data/llama3ft_dpodata_train.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/srv/nfs/home/njnu_zrq/RankCoT/src/modelft/RM_train/llama3_8B")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--zeta", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)

    # LoRA args
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj"])

    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Validation of tokens
    yes_token_id = tokenizer.convert_tokens_to_ids("yes")
    no_token_id = tokenizer.convert_tokens_to_ids("no")
    if accelerator.is_main_process:
        print(f"Token 'yes' ID: {yes_token_id}")
        print(f"Token 'no'  ID: {no_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # 1. Enable input gradients for LoRA + Checkpointing compatibility
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if accelerator.is_local_main_process:
        model.print_trainable_parameters()

    dataset = PairwiseRMDataset(args.data_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=2
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits # [B, L, V]

                B = logits.size(0)
                label_pos = batch["label_indices"] # Index of the 'yes'/'no' token
                
                # FIX: We need the logits of the token *before* the label to predict the label
                # logits[i] predicts input_ids[i+1]
                # To predict input_ids[label_pos], we use logits[label_pos - 1]
                prediction_logits = logits[torch.arange(B, device=logits.device), label_pos - 1] 

                ce_loss = torch.nn.CrossEntropyLoss()(prediction_logits, batch["labels"])

                # Positional Consistency Loss
                # Probability of "yes" and "no"
                prob = torch.softmax(prediction_logits, dim=-1)
                
                # Batch is organized as [ (Query1, Yes), (Query1, No), (Query2, Yes), (Query2, No) ... ]
                # even indices: Prompt asking "Is Chosen > Rejected?" -> Label "yes"
                # odd indices:  Prompt asking "Is Rejected > Chosen?" -> Label "no"
                
                p_yes_first = prob[::2, yes_token_id]   # P("yes" | Chosen > Rejected)
                p_no_second = prob[1::2, no_token_id]   # P("no" | Rejected > Chosen)
                
                # Ideally P("yes" | A>B) should equal P("no" | B>A) -> roughly meaning A>B
                # Or rather, the model's confidence in A>B should be consistent.
                # If model is confident A>B, p_yes_first is high, p_no_second is high.
                # If model thinks B>A, p_yes_first is low, p_no_second is low.
                # So we want them to be close.
                pos_loss = ((p_yes_first - p_no_second) ** 2).mean()

                loss = ce_loss + args.zeta * pos_loss

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                if accelerator.is_local_main_process and global_step % args.logging_steps == 0:
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "ce": f"{ce_loss.item():.4f}",
                        "pos": f"{pos_loss.item():.4e}"
                    })

        if accelerator.is_local_main_process:
            output_path = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(output_path, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("✅ Training completed.")

if __name__ == "__main__":
    main()
import json
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ===================== Configuration =====================
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
TRAIN_JSONL = "./exported_frames/dataset_subtask_train.jsonl"
VAL_JSONL = "./exported_frames/dataset_subtask_val.jsonl"
OUTPUT_DIR = "./weights/system2_subtask_lora"
DEVICE = "mps"  # For Apple Silicon M series

# LoRA hyperparameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training hyperparameters
BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 2
LR = 2e-4
MAX_SEQ_LEN = 1024
# =========================================================

def load_jsonl_to_hf_dataset(jsonl_path):
    # Load jsonl file into Hugging Face Dataset format
    data_list = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data_list.append(item)
    return Dataset.from_list(data_list)


def main():
    # Load multimodal processor for Qwen2‑VL
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def collate_fn(batch):
        # Custom collate function for Qwen2-VL multimodal input
        text_list = []
        image_list = []
        for sample in batch:
            # ✅ 重点修复：content用列表 + type:image，不要手写<image>字符串
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Given the scene image, finish this long‑horizon robot task: {sample['language_instruction']}. Output the sub‑task list as a valid JSON array only."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps(sample["subtask_list"])}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text_list.append(text)
            img = Image.open(sample["image"]).convert("RGB")
            image_list.append(img)

        # Process text + image together
        inputs = processor(
            text=text_list,
            images=image_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN
        )
        # Copy input_ids as labels for causal LM
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    # Use float32, bf16 mixed precision on MPS requires torch>=2.6
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=DEVICE
    )

    # Configure LoRA applied to LLM transformer layers
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    # Print trainable / total parameter statistics
    model.print_trainable_parameters()

    # Load train and validation dataset
    train_ds = load_jsonl_to_hf_dataset(TRAIN_JSONL)
    val_ds = load_jsonl_to_hf_dataset(VAL_JSONL)

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    # Start SFT training
    trainer.train()
    # Save only LoRA adapter weights
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ LoRA adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

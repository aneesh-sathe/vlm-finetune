from datasets import features, load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
import os

ds_id = "aneesh-sathe/operating-systems-vqa-dpo"
dataset = load_dataset(ds_id, split="train")
dataset = dataset.shuffle(seed=42).select(range(100))

model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)

def format_ds(example):
    
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
    chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
    rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
    
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
   

    
    max_size = processor.image_processor.size["longest_edge"] // 2
    example["image"].thumbnail((max_size, max_size))
    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

dataset = dataset.map(format_ds, remove_columns=dataset.column_names, num_proc=os.cpu_count())

f = dataset.features
f["images"] = features.Sequence(features.Image(decode=True))
dataset = dataset.cast(f)

training_args = DPOConfig(
    output_dir="qwen-ft",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    num_train_epochs=1,
    push_to_hub=False,

    dataset_num_proc=os.cpu_count(),
    dataloader_num_workers=os.cpu_count(),
    logging_steps=10,
    )

trainer = DPOTrainer(
    model,
    ref_model=None, 
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor,
    peft_config=LoraConfig(target_modules="all-linear"),
)

trainer.train()




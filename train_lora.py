import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from config import *

# 加载校园微调对话数据集
with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    train_dataset = json.load(f)

# 加载基础模型与分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# LoRA配置
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 训练超参
train_args = TrainingArguments(
    output_dir=LORA_WEIGHT_PATH,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=MAX_TRAIN_EPOCH,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True
)

# SFT训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=train_args,
    dataset_text_field="output"
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(LORA_WEIGHT_PATH)
    print("🎉 LoRA微调训练完成，权重已保存至lora_weights文件夹！")
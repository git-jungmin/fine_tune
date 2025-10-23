from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# ✅ Base model
BASE_MODEL = "Qwen/Qwen2-1.5B"
OUTPUT_DIR = "./lora-ux-qwen2"

# ✅ Dataset
dataset = load_dataset("json", data_files={"train": "./train.jsonl"})

# ✅ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(examples):
    input_texts = []
    for q, a in zip(examples["prompt"], examples["response"]):
        text = f"### 질문: {q}\n### 답변: {a}"
        input_texts.append(text)
    tokens = tokenizer(
        input_texts,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(preprocess, batched=True)

# ✅ LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ✅ 모델 로드 (MPS 호환)
print("🚀 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,          # ✅ M1/M2에서 fp16 대신 bf16 사용
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model = get_peft_model(model, lora_config)

# ✅ 학습 설정
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    bf16=True,                           # ✅ MPS-friendly precision
    save_total_limit=1,
    logging_steps=10,
    save_strategy="epoch",
)

# ✅ Trainer 설정
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# ✅ 학습 시작
print("🔥 Training LoRA adapter...")
trainer.train()

# ✅ 결과 저장
print("💾 Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ LoRA training finished:", OUTPUT_DIR)
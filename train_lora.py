import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using device: {device}")

# 1️⃣ 모델 로드
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🩵 패딩 토큰 없을 때 eos_token으로 대체
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 2️⃣ LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# 3️⃣ 데이터 로드 및 전처리
dataset = load_dataset("json", data_files="dataset.jsonl")["train"]

def format_data(example):
    text = f"### 질문: {example['prompt']}\n### 답변: {example['response']}"
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(format_data, remove_columns=dataset.column_names)

# 4️⃣ 학습 설정
training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none"
)

# 5️⃣ Trainer 실행
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

trainer.train()

# 6️⃣ 결과 저장
model.save_pretrained("./lora-ux")
print("✅ LoRA 어댑터 학습 완료 → ./lora-ux 폴더에 저장되었습니다.")
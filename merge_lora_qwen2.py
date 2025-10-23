from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "Qwen/Qwen2-1.5B"
LORA_DIR = "./lora-ux-qwen2"
OUTPUT_DIR = "./merged-qwen2-1.5b"

print("🚀 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("🔗 Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.merge_and_unload()

print("💾 Saving merged model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Merge complete! Saved to {OUTPUT_DIR}")
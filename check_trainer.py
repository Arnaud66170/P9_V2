from transformers import TrainingArguments
args = TrainingArguments(output_dir="./tmp_test")
print(f"✅ Trainer prêt avec device : {args.device}")
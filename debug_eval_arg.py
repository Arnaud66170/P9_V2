from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./debug_test",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16
)

print("OK : evaluation_strategy =", args.evaluation_strategy)

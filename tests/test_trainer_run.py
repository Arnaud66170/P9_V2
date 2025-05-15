from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(output_dir="./tmp_test")
print(training_args.device)

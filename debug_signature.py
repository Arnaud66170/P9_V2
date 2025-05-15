# debug_signature.py
# from transformers import TrainingArguments
# print(TrainingArguments.__doc__)
from transformers import TrainingArguments

args = TrainingArguments(output_dir="test_output")
print("TrainingArguments loaded avec succès.")

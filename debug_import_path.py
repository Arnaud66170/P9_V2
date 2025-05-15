# debug_import_path.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import transformers.training_args as ta
print("Loaded from:", ta.__file__)


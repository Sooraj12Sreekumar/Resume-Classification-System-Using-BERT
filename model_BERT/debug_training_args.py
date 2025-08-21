import inspect
from transformers import TrainingArguments

print("TrainingArguments defined in:", inspect.getsourcefile(TrainingArguments))
print("TrainingArguments init file:", TrainingArguments.__init__.__code__.co_filename)
print("Init arguments:", TrainingArguments.__init__.__code__.co_varnames)


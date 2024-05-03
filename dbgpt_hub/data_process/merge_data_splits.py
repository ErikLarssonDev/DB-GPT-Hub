TRAIN_DATA="dbgpt_hub/data/example_text2sql_train_one_shot.json"
EVAL_DATA="dbgpt_hub/data/example_text2sql_dev_one_shot.json"
TRAIN_EVAL_DATA="dbgpt_hub/data/example_text2sql_train_dev_one_shot.json"

import json
 
train_data = open(TRAIN_DATA)
eval_data = open(EVAL_DATA)
 
train_dataset = json.load(train_data)
eval_dataset = json.load(eval_data)
 
merged_dataset = {"train": train_dataset, "eval": eval_dataset}

with open(TRAIN_EVAL_DATA, 'w') as fp:
    json.dump(merged_dataset, fp, indent=5)
import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Define a function to parse the log file
def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    easy = []
    medium = []
    hard = []
    extra = []
    all = []
    checkpoint_ids = []
    for line in lines:
        match = re.search(r'checkpoint-(\d+)$', line)
        if match:
            checkpoint_ids.append(int(match.group(1)))
        match_accuracy = re.search(r'execution\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match_accuracy:
            easy.append(float(match_accuracy.group(1)))
            medium.append(float(match_accuracy.group(2)))
            hard.append(float(match_accuracy.group(3)))
            extra.append(float(match_accuracy.group(4)))
            all.append(float(match_accuracy.group(5)))

    return checkpoint_ids, easy, medium, hard, extra, all

def sort_values(values, checkpoint_ids):
    combined = list(zip(values, checkpoint_ids))

    # Sort the combined list based on the second element of each tuple (ids)
    sorted_combined = sorted(combined, key=lambda x: x[0])

    # Extract the sorted values and ids back into separate lists
    return zip(*sorted_combined)


# Path to the log file
log_file_path = '/home/erila/llm-finetune/DB-GPT-Hub/dbgpt_hub/output/logs/pred_20240510_1036.log'

# Parse the log file and get the data
checkpoint_ids, easy, medium, hard, extra, all = parse_log_file(log_file_path)

if len(checkpoint_ids) > len(easy):
    checkpoint_ids = checkpoint_ids[:-1]

# Combine all lists into tuples
combined_lists = list(zip(checkpoint_ids, easy, medium, hard, extra, all))

# Sort the combined list based on the first element of each tuple (ids)
sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])

# Extract the sorted values back into separate lists
checkpoint_ids, easy, medium, hard, extra, all = zip(*sorted_combined_lists)

best_all_idx = np.argmax(all)
print(f"Best checkpoint: {checkpoint_ids[best_all_idx]}")
print(f"Best execution score (easy): {easy[best_all_idx]}")
print(f"Best execution score (medium): {medium[best_all_idx]}")
print(f"Best execution score (hard): {hard[best_all_idx]}")
print(f"Best execution score (extra): {extra[best_all_idx]}")
print(f"Best execution score (all): {all[best_all_idx]}")

# Plotting the data
plt.plot(checkpoint_ids, easy, label="Easy")
plt.plot(checkpoint_ids, medium, label="Medium")
plt.plot(checkpoint_ids, hard, label="Hard")
plt.plot(checkpoint_ids, extra, label="Extra")
plt.plot(checkpoint_ids, all, label="All")
plt.legend()
plt.xlabel('Step (541 steps per epoch)')
plt.ylabel('Execution Accuracy')
plt.title('Execution Accuracy vs Number of Training Steps')
plt.grid(True)
plt.savefig('/home/erila/llm-finetune/DB-GPT-Hub/dbgpt_hub/output/results_llama3-lora-' + log_file_path.split('/')[-1].replace('.log', '.png'))
plt.show()

import random

# Read all lines from the file, skipping empty ones
with open('./tiny_reach_dataset/reach_target_4_objects/metadata.jsonl', 'r') as f:
    lines = [line for line in f if line.strip()]

# Shuffle the list in place
random.shuffle(lines)

# Write the shuffled lines back to the same file (in place)
with open('metadata.jsonl', 'w') as f:
    f.writelines(lines)

import subprocess
import random
from typing import List, Dict, Any
import json
import statistics

NUM_LAYERS = 16
NUM_EXPERTS = 64
BASE_MODEL = "allenai/OLMoE-1B-7B-0924"
EVAL_TASKS = ["mmlu_virology", "mmlu_abstract_algebra"]

TASK_NAME_MAP = {t: t.replace("mmlu_", "") for t in EVAL_TASKS}

def run_evaluation(expert_masks_str: str) -> Dict[str, Any]:
    model_args = f"pretrained={BASE_MODEL},expert_masks={expert_masks_str}"
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "moe-zedong-hf",
        "--model_args", model_args,
        "--tasks", ",".join(EVAL_TASKS),
        "--device", "cuda"
    ]

    print("DEBUG: Running command:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()

    print("DEBUG: stdout from evaluation:\n", stdout)
    print("DEBUG: stderr from evaluation:\n", stderr)

    results = {}
    lines = stdout.split("\n")
    print("DEBUG: Splitting stdout into lines. Number of lines:", len(lines))

    reverse_map = {v: k for k, v in TASK_NAME_MAP.items()}
    print("DEBUG: reverse_map:", reverse_map)

    for i, line in enumerate(lines):
        print(f"DEBUG: Processing line {i}: {repr(line)}")
        
        raw_parts = line.split("|")
        parts = [p.strip() for p in raw_parts if p.strip()]
        print("DEBUG: parts after splitting and stripping:", parts)

        if not parts:
            print("DEBUG: Empty parts, not a task line.")
            continue

        short_task_name = parts[0]
        if short_task_name not in reverse_map:
            print(f"DEBUG: '{short_task_name}' is not in reverse_map, not a known task line.")
            continue

        full_task_name = reverse_map[short_task_name]
        print(f"DEBUG: Matched short task name '{short_task_name}' to full task '{full_task_name}'")

        if len(parts) < 7:
            print("DEBUG: Not enough parts to parse metric and value:", parts)
            continue

        metric = parts[4]
        if metric != 'acc':
            print(f"DEBUG: Unexpected metric '{metric}' in parts: {parts}")
            continue

        try:
            value = float(parts[6])
        except ValueError:
            print(f"DEBUG: Could not convert '{parts[6]}' to float for the main value. parts: {parts}")
            continue

        stderr_value = 0.0
        if len(parts) > 8 and parts[7] == '±':
            try:
                stderr_value = float(parts[8])
            except ValueError:
                print(f"DEBUG: Could not convert '{parts[8]}' to float for stderr. parts: {parts}")

        results[full_task_name] = {
            metric: value,
            "stderr": stderr_value
        }
        print(f"DEBUG: Stored results[{full_task_name}] = {{'acc': {value}, 'stderr': {stderr_value}}}")

    print("DEBUG: Final parsed results:", results)
    return results


def generate_expert_mask_str(mask_dict):
    masks = []
    for layer, experts in mask_dict.items():
        if experts:
            experts_str = "+".join(str(x) for x in sorted(experts))
            masks.append(f"model.layers.{layer}.mlp{{{experts_str}}}")
    return "|".join(masks)

###############################################################################
# Heuristic Search Logic
###############################################################################

baseline_masks = {}
expert_masks_str = generate_expert_mask_str(baseline_masks)
baseline_results = run_evaluation(expert_masks_str)

target_task = "mmlu_virology"
if target_task not in baseline_results:
    raise ValueError(f"Target task {target_task} not found in baseline results. Available: {list(baseline_results.keys())}")

baseline_acc = baseline_results[target_task]["acc"]
print("Baseline accuracy on", target_task, ":", baseline_acc)

expert_impact = []

for layer in range(NUM_LAYERS):
    for expert in range(NUM_EXPERTS):
        mask_dict = {layer: {expert}}
        mask_str = generate_expert_mask_str(mask_dict)
        results = run_evaluation(mask_str)
        if target_task in results:
            acc = results[target_task]["acc"]
            diff = baseline_acc - acc
            expert_impact.append((layer, expert, diff))
            print(f"Layer {layer}, Expert {expert}, diff = {diff}")

expert_impact.sort(key=lambda x: x[2], reverse=True)
top_experts = expert_impact[:10]
print("Top experts impacting target task:", top_experts)

selected_experts = {}
for (l, e, d) in top_experts:
    if l not in selected_experts:
        selected_experts[l] = set()
    selected_experts[l].add(e)

combined_mask_str = generate_expert_mask_str(selected_experts)
combined_results = run_evaluation(combined_mask_str)
combined_acc = combined_results[target_task]["acc"]
combined_diff = baseline_acc - combined_acc
print("Combined mask accuracy drop:", combined_diff)

top_50 = expert_impact[:50]
best_combination = None
best_combination_acc = baseline_acc

for _ in range(10):
    random_subset = random.sample(top_50, k=5)
    mask_dict = {}
    for (l, e, d) in random_subset:
        if l not in mask_dict:
            mask_dict[l] = set()
        mask_dict[l].add(e)
    test_mask_str = generate_expert_mask_str(mask_dict)
    test_results = run_evaluation(test_mask_str)
    test_acc = test_results[target_task]["acc"]
    if test_acc < best_combination_acc:
        best_combination_acc = test_acc
        best_combination = random_subset

print("Best random combination found:", best_combination, "with acc:", best_combination_acc)

with open("expert_impact_results.json", "w") as f:
    json.dump({
        "baseline_results": baseline_results,
        "expert_impact": expert_impact,
        "best_random_combination": best_combination,
        "best_random_combination_acc": best_combination_acc
    }, f, indent=2)

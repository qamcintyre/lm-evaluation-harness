import subprocess
import random
import json
import statistics
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
NUM_LAYERS = 16
NUM_EXPERTS = 64
BASE_MODEL = "allenai/OLMoE-1B-7B-0924"
EVAL_TASKS = ["mmlu_virology", "mmlu_abstract_algebra"]  
TASK_NAME_MAP = {t: t.replace("mmlu_", "") for t in EVAL_TASKS}
ALL_EXPERTS = [(layer, expert) for layer in range(NUM_LAYERS) for expert in range(NUM_EXPERTS)]
TOTAL_EXPERTS = NUM_LAYERS * NUM_EXPERTS

def generate_expert_mask_str(mask_dict):
    """
    Convert a dictionary of {layer: set_of_experts} into the mask string.
    """
    masks = []
    for layer, experts in mask_dict.items():
        if experts:
            experts_str = "+".join(str(x) for x in sorted(experts))
            masks.append(f"model.layers.{layer}.mlp{{{experts_str}}}")
    return "|".join(masks)

def run_evaluation(expert_masks_str: str):
    model_args = f"pretrained={BASE_MODEL},expert_masks={expert_masks_str}"
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "moe-zedong-hf",
        "--model_args", model_args,
        "--tasks", ",".join(EVAL_TASKS),
        "--device", "cuda"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()

    results = {}
    lines = stdout.split("\n")
    reverse_map = {v: k for k, v in TASK_NAME_MAP.items()}

    for line in lines:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts:
            continue
        short_task_name = parts[0]
        if short_task_name not in reverse_map:
            continue
        full_task_name = reverse_map[short_task_name]

        if len(parts) < 7:
            continue
        metric = parts[4]
        if metric != 'acc':
            continue
        try:
            value = float(parts[6])
        except ValueError:
            continue

        stderr_value = 0.0
        if len(parts) > 8 and parts[7] == '±':
            try:
                stderr_value = float(parts[8])
            except ValueError:
                pass

        results[full_task_name] = {
            metric: value,
            "stderr": stderr_value
        }

    return results

baseline_results = run_evaluation("")
baseline_scores = {task: baseline_results.get(task, {"acc": 0})["acc"] for task in EVAL_TASKS}

percentages = [0.0, 0.1, 0.2, 0.3]  # 0%, 10%, 20%, 30% experts masked
num_repeats = 5  # number of random subsets per percentage for error bars

# results_dict[task][percentage] = list of accuracies from different random runs
results_dict = {task: {pct: [] for pct in percentages} for task in EVAL_TASKS}

for pct in percentages:
    if pct == 0.0:
        for task in EVAL_TASKS:
            for _ in range(num_repeats):
                results_dict[task][pct].append(baseline_scores[task])
        continue

    num_to_mask = int(pct * TOTAL_EXPERTS)
    for _ in range(num_repeats):
        chosen = random.sample(ALL_EXPERTS, k=num_to_mask)
        mask_dict = {}
        for (l, e) in chosen:
            if l not in mask_dict:
                mask_dict[l] = set()
            mask_dict[l].add(e)

        mask_str = generate_expert_mask_str(mask_dict)
        res = run_evaluation(mask_str)

        for task in EVAL_TASKS:
            acc = res.get(task, {"acc": 0})["acc"]
            results_dict[task][pct].append(acc)

plot_data = []
for task in EVAL_TASKS:
    for pct in percentages:
        accs = results_dict[task][pct]
        mean_acc = statistics.mean(accs) if accs else 0
        std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0
        plot_data.append({
            "task": task,
            "percentage": pct,
            "mean_acc": mean_acc,
            "std_acc": std_acc
        })

import pandas as pd
df = pd.DataFrame(plot_data)

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x="percentage", y="mean_acc", hue="task", marker="o",
             err_style="bars", err_kws={"capsize": 5}, 
             ci=None)  # ci=None since we manually have std, we could also use errorbar='se'

for task in EVAL_TASKS:
    subset = df[df["task"] == task]
    plt.errorbar(subset["percentage"], subset["mean_acc"], yerr=subset["std_acc"],
                 fmt='none', ecolor='gray', capsize=5)

plt.title("Impact of Random Expert Masking on MMLU Tasks")
plt.xlabel("Fraction of Experts Masked")
plt.ylabel("Accuracy")
plt.legend(title="Task")
plt.tight_layout()

plt.savefig("expert_masking_experiment.png")

with open("random_expert_masking_results.json", "w") as f:
    json.dump({
        "baseline_scores": baseline_scores,
        "results": results_dict,
        "aggregated": plot_data
    }, f, indent=2)

plt.show()

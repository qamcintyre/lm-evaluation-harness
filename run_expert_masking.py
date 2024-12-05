import random
import subprocess

def generate_random_expert_masks(num_layers=16, num_experts=64, mask_fraction=0.10):
    masks = []
    num_to_mask = int(num_experts * mask_fraction)
    
    for layer in range(num_layers):
        masked_experts = set(random.sample(range(num_experts), num_to_mask))
        experts_str = "+".join(str(x) for x in masked_experts)
        masks.append(f'model.layers.{layer}.mlp{{{experts_str}}}')
    
    return "|".join(masks)

masks = generate_random_expert_masks()
model_args = f"pretrained=allenai/OLMoE-1B-7B-0924,expert_masks={masks}"

cmd = [
    "python", "-m", "lm_eval",
    "--model", "moe-zedong-hf",
    "--model_args", model_args,
    "--tasks", "mmlu_virology",
    "--device", "cuda"
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd)
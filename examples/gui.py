import torch

from simple_generation import SimpleGenerator

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
generator = SimpleGenerator(model_name, torch_dtype=torch.bfloat16, device="cuda:1")

generator.gui(
    do_sample=True,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.85,
    top_k=50,
)

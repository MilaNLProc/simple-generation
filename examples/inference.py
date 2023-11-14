from pprint import pprint

import torch

from simple_generation import SimpleGenerator

model_name = "meta-llama/Llama-2-7b-chat-hf"
generator = SimpleGenerator(model_name, torch_dtype=torch.bfloat16)

texts = [
    "Write a poem in the style of Leonardo Da Vinci.",
    "Tell me what's 2 + 2.",
    "Write ten facts about the moon.",
] * 128
texts = texts

gen = generator(
    texts,
    skip_prompt=True,
    do_sample=True,
    max_new_tokens=256,
    temperature=0,
    top_p=0.05,
    top_k=50,
    starting_batch_size=16,
)

print("Generated texts:", generator.local_rank, len(gen))
pprint(gen[:5])

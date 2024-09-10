from pprint import pprint

import torch

from simple_generation import SimpleGenerator

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
generator = SimpleGenerator(
    model_name, engine="vllm", dtype=torch.bfloat16, tokenizer_mode="auto"
)

texts = (
    [
        "Write a poem in the style of Leonardo Da Vinci.",
    ]
    * 32
    + [
        "Tell me what's 2 + 2.",
    ]
    * 32
    + [
        "Write ten facts about the moon.",
    ]
    * 32
)

gen = generator(
    texts,
    max_tokens=256,
    temperature=0.8,
    top_p=0.85,
    top_k=50,
    apply_chat_template=True,
    add_generation_prompt=True,
)

assert len(gen) == len(texts)

print("Generated texts:", generator.local_rank, len(gen))
pprint(gen[:5])

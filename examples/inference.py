from pprint import pprint

import torch

from simple_generation import SimpleGenerator

# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
generator = SimpleGenerator(model_name, torch_dtype=torch.bfloat16, device="cuda:5")

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

print(len(texts))

gen = generator(
    texts,
    skip_prompt=True,
    do_sample=True,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.85,
    top_k=50,
    starting_batch_size=16,
    apply_chat_template=True,
)

assert len(gen) == len(texts)

print("Generated texts:", generator.local_rank, len(gen))
pprint(gen[:5])

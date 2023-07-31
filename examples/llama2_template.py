from simple_generation import SimpleGenerator
import torch
from pprint import pprint

model_name = "meta-llama/Llama-2-7b-chat-hf"
generator = SimpleGenerator(
    model_name, system_prompt="llama-2", torch_dtype=torch.bfloat16
)

texts = [
    "Write a poem in the style of Leonardo Da Vinci",
    "Tell me what's 2 + 2.",
    "Write ten facts about the moon.",
]
gen = generator(
    texts,
    return_full_text=False,
    do_sample=True,
    max_new_tokens=256,
    temperature=0.1,
    top_p=0.05,
    top_k=50,
)

print(gen)

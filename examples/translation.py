"""
Example: Machine Translation

This script contains the minimum code to translate the En->It split of the EuroParl dataset
using a Opus MT model and Simple Generation.
"""
import torch
from datasets import load_dataset
from pprint import pprint

from simple_generation import SimpleGenerator

model_name = "Helsinki-NLP/opus-mt-en-it"
generator = SimpleGenerator(model_name, torch_dtype=torch.bfloat16)

dataset = load_dataset("europarl_bilingual", lang1="en", lang2="it")["train"].select(
    range(1000)  # remove .select(range(1000)) to translate the full split
)
texts = [sample["translation"]["en"] for sample in dataset]
references = [sample["translation"]["it"] for sample in dataset]

pprint("Some texts:")
pprint(texts[:3])

output = generator(
    texts,
    skip_prompt=False,
    num_beams=5,
    max_new_tokens=256,
    starting_batch_size=128,
)

for t, r, tr in zip(texts[:3], references[:3], output[:3]):
    print(f"Source: {t}")
    print(f"Reference: {r}")
    print(f"Translation: {tr}")
    print("####")

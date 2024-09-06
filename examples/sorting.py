from pprint import pprint

import torch
import time
from simple_generation import SimpleGenerator

# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
generator = SimpleGenerator(model_name, torch_dtype=torch.bfloat16, device="cuda")

short = "Tell me what's 2 + 2."
long = """Summarize the following passage: 
Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ,[9] is an American businessman and former professional basketball player. He played 15 seasons in the National Basketball Association (NBA) between 1984 and 2003, winning six NBA championships with the Chicago Bulls. He was integral in popularizing basketball and the NBA around the world in the 1980s and 1990s,[10] becoming a global cultural icon.[11] His profile on the NBA website states, "By acclamation, Michael Jordan is the greatest basketball player of all time."[12]"""

texts = [short] * 3 + [long] * 5
# replicate the list 128 times
texts = texts * 128

print(len(texts))

stime = time.time()

gen_args = {
    "skip_prompt": True,
    "do_sample": True,
    "max_new_tokens": 256,
    "temperature": 0.8,
    "top_p": 0.85,
    "top_k": 50,
    "starting_batch_size": 16,
    "apply_chat_template": True,
}

responses = generator(texts, **gen_args)
assert len(responses) == len(texts)
time_no_sorting = time.time() - stime
print(responses[:5])

print("Redoing the generation with sorting")
gen_args["sort_prompts_by_length"] = True
stime = time.time()
responses = generator(texts, **gen_args)
assert len(responses) == len(texts)
time_with_sorting = time.time() - stime

print("Time without sorting:", time_no_sorting)
print("Time with sorting:", time_with_sorting)
print("Speedup:", time_no_sorting / time_with_sorting)

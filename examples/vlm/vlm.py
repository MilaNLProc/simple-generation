import requests
from simple_generation.vlm import SimpleVLMGenerator
import fire
import time
from PIL import Image
import copy
import torch
from secrets import token_hex
import pandas as pd


def main(model_name_or_path: str = "llava-hf/llava-v1.6-mistral-7b-hf", n: int = 32):
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    images = [copy.deepcopy(image) for _ in range(n)]
    prompts = ["[INST] <image>\nWhat's in this image? [/INST]"] * len(images)

    generator = SimpleVLMGenerator(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        # device="cuda",
        device_map="auto",
        # attn_implementation="flash_attention_2",
    )

    responses = generator(
        prompts,
        images,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        skip_prompt=True,
        batch_size=8,
        num_workers=2,
    )

    df = pd.DataFrame(dict(prompt=prompts, response=responses))
    df.to_csv(f"./test_{token_hex(4)}.tsv", sep="\t")


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Elapsed {time.time() - stime} seconds")

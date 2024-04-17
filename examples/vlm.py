import requests
from simple_generation.vlm import SimpleVLMGenerator
import fire
import time
from PIL import Image
import copy

def main(n: int = 32):
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    images = [copy.deepcopy(image) for _ in range(n)]

    generator = SimpleVLMGenerator(
        
    )



if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Elapsed {time.time() - stime} seconds")

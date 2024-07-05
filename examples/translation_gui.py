import torch
from transformers import QuantoConfig
from simple_generation import SimpleGenerator

model_name = "Unbabel/TowerInstruct-7B-v0.2"

quant_config = QuantoConfig(weights="int4", activations="int8")
generator = SimpleGenerator(
    model_name, torch_dtype=torch.bfloat16, quant_config=quant_config
)

generator.gui(type="translation", do_sample=False, max_new_tokens=256, num_beams=5)

# Simple Generation

Simple Generator offers a minimal interface to text generation with hugginface models. The core idea is to ship many neat features out of the box and avoid boilerplate code.

This is mainly for personal use.

## Features

- any model that can be loaded with `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
- loading with 8bit or 4bit quantization
- batched inference using `datasets` and `tqdm`
- carbon emission estimates using `codecarbon`
- auto gpus placement and inference
- prefix addition
- only torch checkpoints are supported

## Install

```bash
pip install git+https://github.com/g8a9/simple-generation.git
```

## Minimal Example

```python
from simple_generation import SimpleGenerator

model_name = "google/flan-t5-xxl"
gen = SimpleGenerator(model_name, load_in_4bit=True)

texts = [
    "Today is a great day to run a bit. Translate this to Spanish?",
    "Today is a great day to run a bit. Translate this to German?",
]
responses = gen(texts, max_new_tokens=128, do_sample=False, num_beams=4)
```
# Simple Generation

Simple Generator offers a minimal interface to text generation with hugginface models. The core idea is to ship many neat features out of the box and avoid boilerplate code.

This is mainly for personal use and for simple hardware setups (ideally, single-gp).

## Features

- any model that can be loaded with `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
- batched inference for speed
- auto find best batch size
- load and attach LoRA weights
- loading models with 8bit or 4bit quantization 
- carbon emission estimates using `codecarbon`
- auto gpus placement and inference
- prefix addition (`prefix=`)
- return only the generated text (`return_full_text=False`)

### WIP

- auto find the best device placement for speed
- make emission tracking optional and logs less invasive

### What Is Not Supported

- Frameworks other than torch
- Models not in the Huggingface Hub

## Install

```bash
pip install git+https://github.com/g8a9/simple-generation.git
```

## Minimal Example

```python
from simple_generation import SimpleGenerator

model_name = "google/flan-t5-xxl"
gen = SimpleGenerator(model_name, load_in_8bit=True)

texts = [
    "Today is a great day to run a bit. Translate this to Spanish?",
    "Today is a great day to run a bit. Translate this to German?",
]
responses = gen(texts, max_new_tokens=128, do_sample=False, num_beams=4)
```

The script will generate a `emissions.csv` file with estimated emissions.

### LoRA example

```python
gen = SimpleGenerator(
    model_name_or_path="yahma/llama-7b-hf",
    lora_weights="tloen/alpaca-lora-7b",
    load_in_8bit=True,
)

texts = [
    "Write a recipe where the first step is to preheat the oven to 350 degrees.",
    "What is 2 + 2?",
    "What is the capital of France?",
    "There are 5 apples and 3 oranges in a basket. How many pieces of fruit are in the basket?",
    "There are 5 apples and 3 oranges in a basket. How many pieces of fruit are in the basket? Let's think step by step.",
]
responses = gen(texts, max_new_tokens=256, do_sample=True, num_beams=1, batch_size="auto")
```

This code will, in sequence:
- load a base llama model in 8bit
- attach Alpaca LoRA weights to it
- run inference on the given texts finding the largest batch size fitting the available resources

### Prefix example

```python
responses = gen(
    texts,
    prefix="Translate English to Spanish:",
    max_new_tokens=256,
    do_sample=False,
    num_beams=1,
    top_k=50,
    batch_size="auto",
)
```

If you specify a `prefix`, it will be automatically appended to every input text.


## Generation Defaults

If not specified we set some sensible defaults for text generation. **Please note that they might not fit your use case.**

```python
@dataclasses.dataclass
class DefaultGenerationConfig(GenerationConfig):
    """Default generation config.

    We apply this parameters to any .generate() call, unless they are not overridden.
    """
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    temperature: float = 0.75
    top_k: int = 50
    top_p: float = 0.95
    typical_p: float = 1.0
    repetition_penalty: float = 1.1
    num_return_sequences: int = 1
    penalty_alpha: float = 0.2
    length_penalty: int = 1.2
```

## Warning

There seem to be instabilities while using 4bit quantization (not related to this library). Use it only if strictly necessary.

# Simple Generation

Simple Generator offers a minimal interface to run text generation with HuggingFace checkpoint.
The core idea is to ship many neat features out of the box and avoid boilerplate.

This is mainly for personal use and simple hardware setups (ideally, single-node single- or multi-gpu). A good part of it is WIP. \\
Moreover, please note that **the library will apply some (sensible) defaults (on where to place models, generation configuration, and so on) which might not suit your use case** and should be edited accordingly. Please head to [defaults](#defaults) to see a list of things you should be aware of.

Install with:
```bash
pip install git+https://github.com/MilaNLProc/simple-generation.git
```

## Features

- any model that can be loaded with `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
- batched inference for speed (`batch_size=256`)
- auto find best batch size (`batch_size="auto"`, `starting_batch_size=512`)
- torch.compile the model for speed (`compile_model=True`)
- load and attach LoRA weights (`lora_weights=...`)
- system prompt templates for modern chat models (`system_prompts="llama-2"`) using [FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py)
- carbon emission estimates using [codecarbon](https://mlco2.github.io/codecarbon/)
- sparsity and fused kernels for speed with [optimum](https://huggingface.co/docs/optimum/main/en/index) (`use_bettertransformer=True`)

**Loading a Model**

```python
from simple_generation import SimpleGenerator
model_name = "meta-llama/Llama-2-7b-chat-hf"
generator = SimpleGenerator(model_name, load_in_8bit=True)
```

Any named argument to `SimpleGenerator` will be passed the `from_pretrained` HF method. For example, you can
- load models with 8bit or 4bit quantization (`load_in_[4|8]bit=True`)
- set torch dtypes (`torch_dtype=torch.bfloat16`)
- use auto GPUs placement and inference (`device_map="auto"`)

**Running Inference**

```python
texts = [
    "Write a poem in the style of Leonardo Da Vinci",
    "Tell me what's 2 + 2.",
    "Translate the following sentence to Spanish: I went to the supermarket to buy a lighter."
]
responses = generator(texts)
```

The `__call__` function accepts several named arguments (see examples below). For example:

- return only the generated text (`skip_prompt=True`)
- periodic logging of decoded samples (`log_batch_sample=`)

### WIP

- auto find the best device placement for speed
- efficient duplicate and quasi-duplicate removal
- ~~support system prompt chat formatting following standard templates (e.g., Vicuna, LLaMA 2)~~
- support auto gptq quantized models and tentatively GGML
- spawn web app to quickly local test conversation with gradio

### What Is Not Supported

- frameworks other than torch
- models not in the Huggingface Hub
- example-specific decoding parameters (i.e., given a batch of samples passed to the `__call__`, we will apply the same set of parameters for every sample)

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


## Defaults

If not specified we set some sensible defaults. **Please note that they might not fit your use case.**

**Default Generation Configuration**

- If not specified otherwise, the library will use the generation configs listed below, **overriding the default config loaded from the specified model**.
```python
@dataclasses.dataclass
class DefaultGenerationConfig(GenerationConfig):
    """Default generation config.

    We apply this parameters to any .generate() call, unless they are not overridden.
    """
    max_new_tokens: int = 512
    do_sample: bool = True  # set to False for greedy decoding
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    num_return_sequences: int = 1
```

- We set the tokenizer to use left padding. This is required for batched inference with `AutoModelForCausalLM` but should also be fine with any other `AutoModelForSeq2SeqLM` since we use attention masks.


## Warning

There seem to be instabilities while using 4bit quantization (not related to this library). Use it only if strictly necessary.

## Reference

```bibtex
@misc{milanlp-2023-simple-generation,
  author = {Giuseppe Attanasio},
  title = {{S}imple {G}eneration},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/MilaNLProc/simple-generation}}
}
```

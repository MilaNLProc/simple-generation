![Simple Generation Dining](/docs/_static/banner.png)

[![Latest PyPI version](https://img.shields.io/pypi/v/simple-generation.svg)](https://pypi.python.org/pypi/simple-generation)
[![Documentation Status](https://readthedocs.org/projects/simple-generation/badge/?version=latest)](https://simple-generation.readthedocs.io/en/latest/?version=latest)
[![downloads badge](https://pepy.tech/badge/simple-generation/month)](https://pepy.tech/project/simple-generation)

# Simple Generation

Simple Generation offers a minimal interface to run text generation with HuggingFace checkpoints.
The core idea is to ship many neat features out of the box, avoiding boilerplate.

*Simplegen is mainly for personal use and simple hardware setups (ideally, single-node single- or multi-gpu). If you are looking for production-ready inference engine consider looking elsewhere :) )*

Moreover, please note that **the library will apply some (sensible) defaults (on where to place models, generation configuration, and so on) which might not suit your use case** and should be edited accordingly. Please head to [defaults](#defaults) to see a list of things you should be aware of.

Install:
```bash
pip install simple-generation
```

We support additional functionalities such as inference with Vision Language Models (VLMs) and the creation of quick web interfaces to interact with models (GUIs). Install them with `vlm|gui|all`:
```bash
pip install simple-generation[vlm]

# or
pip install simple-generation[gui]

# or
pip install simple-generation[all]
```

## Features

- generate with any model that can be loaded with `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
- batched inference for speed (`batch_size=256`)
- auto find the largest batch size fitting in your accelerator (`batch_size="auto"`, `starting_batch_size=512`)
- torch.compile the model for speed (`compile_model=True`)
- load and attach LoRA weights (`lora_weights=...`)
- chat templates for modern chat models (`apply_chat_template=True` in `__call__`)
- carbon emission estimates using [codecarbon](https://mlco2.github.io/codecarbon/)
- sparsity and fused kernels for speed with [optimum](https://huggingface.co/docs/optimum/main/en/index) (`use_bettertransformer=True`)
- DDP for single-node, multi-gpu setups using [accelerate](https://github.com/huggingface/accelerate). See [Distributed Inference](#distributed-inference)
- GUI for quick interaction with models. See [GUI](#GUI)

**Vision-Language Models**

- all of the above for [LLaVA](https://huggingface.co/docs/transformers/main/en/model_doc/llava#overview),
[IDEFICS](https://huggingface.co/docs/transformers/main/en/model_doc/idefics#overview), 
[IDEFICS2](https://huggingface.co/docs/transformers/v4.40.1/en/model_doc/idefics2),
and [BLIP](https://huggingface.co/docs/transformers/main/en/model_doc/blip#overview).

For an example look into `./examples/vlm`.

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
responses = generator(texts, apply_chat_template=True, add_generation_prompt=True)
```

The `__call__` function accepts several named arguments (see examples below). For example:

- return only the generated text (`skip_prompt=True`)
- periodic logging of decoded samples (`log_batch_sample=`)

### WIP

- auto find the best device placement for speed
- efficient duplicate and quasi-duplicate removal
- ~~support system prompt chat formatting following standard templates (e.g., Vicuna, LLaMA 2)~~
- support auto gptq quantized models and tentatively GGML
- ~~spawn web app to quickly local test conversation with gradio~~
- even faster inference engine with [vllm](https://vllm.ai/)
- ~~distributed inference for single-node, multi-gpu setups~~

### What Is Not Supported

- frameworks other than `torch`
- models not in the Huggingface Hub
- example-specific decoding parameters (i.e., given a batch of samples passed to the `__call__`, we will apply the same set of parameters for every sample)

## Examples

### Getting Started

```python
from simple_generation import SimpleGenerator

model_name = "google/flan-t5-xxl"
gen = SimpleGenerator(model_name, load_in_8bit=True)

texts = [
    "Today is a great day to run a bit. Translate this to Spanish?",
    "Today is a great day to run a bit. Translate this to German?",
]
responses = gen(texts, max_new_tokens=128, do_sample=False, num_beams=4, skip_prompt=False)
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

### Chat Templates

Starting from v0.2.0, we leverage Hugging Face's [chat templating system](https://huggingface.co/docs/transformers/chat_templating). You can activate it by using `apply_chat_template=True` when invoking `__call__`.
You can also enable the generation prompt by setting `add_generation_prompt=True`. See [here](https://huggingface.co/docs/transformers/chat_templating#what-are-generation-prompts) why that might be a good idea.

**Note: chat templates are not enabled by default!**

### GUI

When installed, Simple Generation exposes a basic command line interface (`simplegen`) to fire up quick web apps.

We currently support:

- Chat Interface: by calling `simplegen chat`. Note that, since **everything will run locally**, you can fill up your VRAM quite easily with long chats. Keep an eye on your VRAM usage and clean the chat frequently -- you might notice that memory does not get freed up immediately, but cleaning the chat will reuse the already allocated memory for new chats.
- Traslation Interface: by calling `simplegen translation`. Note that for Opus MT models the choice of source and target languages is not available/

Example
```bash
simplegen chat -m google/gemma-2-9b-it
```

### Simple Translation

This is the minimum code to translate the En->It EuroParl split using a Opus MT neural model.

```python
generator = SimpleGenerator("Helsinki-NLP/opus-mt-en-it")

dataset = load_dataset("europarl_bilingual", lang1="en", lang2="it")["train"]
texts = [sample["translation"]["en"] for sample in dataset]
references = [sample["translation"]["it"] for sample in dataset]

output = generator(
    texts,
    skip_prompt=False,
    num_beams=5,
    max_new_tokens=256,
    starting_batch_size=128,
)
```

### Distributed Inference

Simple Generation supports DDP to run distributed inference in Single-Node, Multi-GPUs setups. Note that a copy of the model will be instantiated in each GPU (instead of smart weights placements across multiple GPUs with `device_map="auto"`), so **each of your GPU will need to have enough space to fit a copy of the model**.

The only change you'll need to take is launching your script with `accelerate`. E.g.,:

```bash
accelerate launch --num_processes 2 examples/inference.py # uses 2 GPUs
```

Note: if you do not specify `--num_processes` all local GPUs will be used.

Some timing tests on 2xA5000 with Llama-2-7b-chat and 384 input prompts:

```shell
# single GPU
CUDA_VISIBLE_DEVICES=0 python examples/inference.py
>> 217s

# two GPUs, smart placement
CUDA_VISIBLE_DEVICES=0,1 python examples/inference.py
>> 219s

# two GPUs, DDP
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 examples/inference.py
>> 105s
```

### Multiple-Request Conversation

The library supports creating a conversation by prompting models with multiple requests. I.e., it is possible to build a multi-turn conversation with fixed user requests. You can use the `conversation_from_user_prompts()` method, that accepts the same arguments of `__call__`.

For example:
```python
from simple_generation import SimpleGenerator
import torch

texts = [
    "What kind of noises did dinosaurs make?",
    "What is the most popular programming language?",
    "What is 2 + 2?",
    "Tell me how to make a cake.",
]

generator = SimpleGenerator(
    "lmsys/vicuna-7b-v1.3",
    load_in_8bit=True,
    torch_dtype=torch.bfloat16,
)

conversation = generator.conversation_from_user_prompts(
    texts,
    do_sample=True,
    top_p=0.95,
    temperature=0.1,
    max_new_tokens=512,
)

print("Conversation:")
print(conversation)
```

will print:

```
Conversation:
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What kind of noises did dinosaurs make? ASSISTANT: It is difficult to know for sure what noises dinosaurs made, as they lived millions of years ago and did not leave behind any recorded sounds. However, scientists can make educated guesses about what some dinosaurs might have sounded like based on their physical characteristics and behavior.

Some scientists have suggested that certain dinosaurs, such as birds, may have made sounds similar to those of modern birds. For example, some scientists believe that the ancestors of modern birds may have made a variety of sounds, including calls, chirps, and trills.

Other scientists have suggested that some dinosaurs, such as theropod dinosaurs, may have made sounds similar to those of modern reptiles, such as hissing or grunting sounds.

It is also possible that some dinosaurs may have made sounds that are not similar to any sounds made by modern animals. For example, some scientists have suggested that certain dinosaurs, such as some of the large herbivorous dinosaurs, may have made sounds similar to those of modern elephants or rhinoceroses.

Overall, it is difficult to know for sure what noises dinosaurs made, but scientists can make educated guesses based on their physical characteristics and behavior.</s>USER: What is the most popular programming language? ASSISTANT: It is difficult to determine the most popular programming language, as different programming languages are popular for different purposes and in different contexts. Some programming languages are more popular for certain types of applications, while others are more popular for others.

For example, Python is a popular programming language for data science and machine learning, while Java is a popular programming language for developing enterprise-level applications. C++ is a popular programming language for developing operating systems and other low-level systems software, while JavaScript is a popular programming language for developing web applications.

Overall, the popularity of a programming language can depend on a variety of factors, including the needs of the project, the skills and preferences of the developer, and the availability of resources and support for the language.</s>USER: What is 2 + 2? ASSISTANT: 2 + 2 is equal to 4. This is a basic arithmetic calculation, and the answer is not dependent on any specific context or situation.</s>USER: Tell me how to make a cake. ASSISTANT: Here is a basic recipe for making a cake:

Ingredients:

* 1 1/2 cups all-purpose flour
* 1 cup granulated sugar
* 1/2 cup unsalted butter, softened
* 2 large eggs
* 1 teaspoon baking powder
* 1/2 teaspoon salt
* 1 cup milk
* 1/2 cup vegetable oil

Instructions:

1. Preheat the oven to 350°F (175°C). Grease and flour a 9-inch (23 cm) round cake pan.
2. In a medium bowl, whisk together the flour, baking powder, and salt.
3. In a large mixing bowl, beat the softened butter and sugar together until light and fluffy.
4. Beat in the eggs one at a time, then stir in the flour mixture until just combined.
5. Stir in the milk and vegetable oil until smooth.
6. Pour the batter into the prepared cake pan and smooth the top.
7. Bake for 30-35 minutes, or until a toothpick inserted into the center of the cake comes out clean.
8. Remove the cake from the oven and let it cool in the pan for 10 minutes. Then, remove the cake from the pan and let it cool completely on a wire rack.

This is a basic recipe for a cake, and there are many variations and modifications that can be made to suit your preferences and the occasion for which you are making the cake. For example, you can add flavorings such as vanilla extract or chocolate chips, or use a different type of flour or sugar if you have a specific dietary need or preference.</s></s>
```

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

- We set the tokenizer to use left padding. This is required for batched inference with `AutoModelForCausalLM` but should also be fine with any other `AutoModelForSeq2SeqLM` since we use attention masks. It is also recommended for VLM batch generations. See [this issue](https://github.com/huggingface/transformers/issues/3021#issuecomment-1454266627) and [usage tips](https://huggingface.co/docs/transformers/model_doc/llava_next#usage-tips) for more details.

## Warning

There seem to be instabilities while using 4bit quantization (not related to this library). Use it only if strictly necessary.

## Acknowledgments

Thanks to Paul Röttger for the many inputs and priceless bug hunting.

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

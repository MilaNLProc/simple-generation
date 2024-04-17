import PIL.Image
import torch
import torch.distributed as dist

import torch.utils
import torch.utils.data
from transformers import (
    GenerationConfig,
    AutoProcessor,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    IdeficsForVisionText2Text,
    Blip2ForConditionalGeneration,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import find_executable_batch_size
from codecarbon import track_emissions
import PIL
import dataclasses
from typing import List, Union, Dict
from tqdm import tqdm
from enum import Enum
from .config import IdeficsHelper
import math
import numpy as np
from datasets import Dataset

logger = get_logger(__name__)

inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


class VLMType(Enum):
    LLAVA = "LLAVA"
    IDEFICS = "IDEFICS"
    QWEN = "QWEN"
    BLIP2 = "BLIP2"


class VLMDataset(torch.utils.data.Dataset):
    def __init__(self, processor, prompts: List[str], images) -> None:
        self.processor = processor
        self.prompts = prompts
        self.images = images

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index) -> Dict:
        pass


# @dataclasses.dataclass
# class DefaultGenerationConfig(GenerationConfig):
#     """Default generation configuration.

#     We apply this parameters to any .generate() call, unless they are not overridden.

#     Attributes:
#         max_new_tokens (int): The maximum number of tokens to generate. Defaults to 512.
#         do_sample (bool): Whether to use sampling or greedy decoding. Defaults to True.
#         temperature (float): The sampling temperature. Defaults to 0.7.
#         top_p (float): The cumulative probability for sampling from the top_p distribution. Defaults to 1.0.
#         top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to 50.
#         num_return_sequences (int): The number of independently computed returned sequences for each element in the batch. Defaults to 1.

#     """

#     max_new_tokens: int = 512
#     do_sample: bool = True
#     temperature: float = 0.7
#     top_p: float = 1.0
#     top_k: int = 50
#     num_return_sequences: int = 1


class SimpleVLMGenerator:
    @property
    def is_ddp(self):
        """Returns True if the model is distributed."""
        return dist.is_available() and dist.is_initialized()

    def __init__(self, model_name_or_path, **model_kwargs):
        self.model_name_or_path = model_name_or_path

        # Per-Model configuration
        if "llava" in self.model_name_or_path.lower():
            model_cls = LlavaNextForConditionalGeneration
            self.vlm_type = VLMType.LLAVA
            # self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        elif "idefics" in self.model_name_or_path.lower():
            model_cls = IdeficsForVisionText2Text
            self.vlm_type = VLMType.IDEFICS
            # self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        elif "blip" in self.model_name_or_path.lower():
            model_cls = Blip2ForConditionalGeneration
            self.vlm_type = VLMType.BLIP2
            # self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        elif "qwen" in self.model_name_or_path.lower():
            raise NotImplementedError()
            model_cls = AutoModelForCausalLM
            self.vlm_type = VLMType.QWEN
            self.processor = AutoTokenizer.from_pretrained(self.model_name_or_path)
        else:
            raise NotImplementedError(f"We do not wrap {model_name_or_path} yet!")

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        # padding_size="left" is required for autoregressive models, and should not make a difference for every other model as we use attention_masks. See: https://github.com/huggingface/transformers/issues/3021#issuecomment-1454266627 for a discussion on why left padding is needed on batched inference
        # This is also relevant for VLM batched generation: https://huggingface.co/docs/transformers/model_doc/llava_next#usage-tips
        self.processor.tokenizer.padding_side = "left"

        # Use accelerator to distribute model if DDP is enabled
        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        user_request_move_to_device = False

        if "device" in model_kwargs:
            logger.info(f"Setting device to {self.device} per user's request.")
            self.device = model_kwargs.pop("device")
            user_request_move_to_device = True

        # try:
        #     self.generation_config = GenerationConfig.from_pretrained(
        #         model_name_or_path
        #     )
        # except Exception as e:
        #     logger.warning("Could not load generation config. Using default one.")
        #     self.generation_config = DefaultGenerationConfig()

        self.model = model_cls.from_pretrained(model_name_or_path, **model_kwargs)

        if self.is_ddp or user_request_move_to_device:
            self.model.to(self.device)
            logger.debug(f"Moving model to {self.device}")

        self.model.eval()

        print(
            f"""
            Simple Generation (VLM) initialization completed!

            Model:
            - id: {self.model_name_or_path},
            - VLM type: {self.vlm_type}

            Model placement:
            - device_map: {model_kwargs.pop('device_map', None)},
            - device: {self.device},

            DDP:
            - distributed inference: {self.is_ddp}

            **Note** We do not enforce input formatting. Be sure to format you inputs according to the used model. See here examples for LlaVA: https://huggingface.co/docs/transformers/model_doc/llava_next#usage-tips
            """
        )

    @track_emissions(log_level="error", measure_power_secs=60)
    @inference_decorator()
    def __call__(
        self,
        texts,
        images,
        batch_size="auto",
        starting_batch_size=256,
        num_workers=4,
        skip_prompt=False,
        # log_batch_sample=-1,
        show_progress_bar=None,
        # prepare_prompts=False,  # keeping it here for consistency
        # apply_chat_template=False,
        # add_generation_prompt=False,
        macro_batch_size: int = 512,
        **generation_kwargs,
    ):
        """
        v2:
        - the user needs to take care of the input format
        - support only single-turn inference, text and images at the same position in the input will be considered an input pair.
        """

        if not isinstance(texts, list):
            logger.debug("Texts is not a list. Wrapping it in a list.")
            texts = [texts]
        if not isinstance(images, list):
            logger.debug("Images is not a list. Wrapping it in a list.")
            images = [images]

        if len(texts) != len(images):
            raise ValueError("Prompt and image counts must be the same.")

        # Prepare model specific processor and generation args
        processor_args = dict()
        if self.vlm_type == VLMType.IDEFICS:
            processor_args["add_end_of_utterance_token"] = False

            exit_condition = self.processor.tokenizer(
                "<end_of_utterance>", add_special_tokens=False
            ).input_ids
            generation_kwargs["eos_token_id"] = exit_condition
            bad_words_ids = self.processor.tokenizer(
                ["<image>", "<fake_token_around_image>"], add_special_tokens=False
            ).input_ids
            generation_kwargs["bad_words_ids"] = bad_words_ids

        n_macro_batches = math.ceil(len(images) / macro_batch_size)
        iterator = np.array_split(zip(texts, images), n_macro_batches)
        responses = list()
        for curr_prompts, curr_images in tqdm(
            iterator, desc="Macro batch", total=n_macro_batches
        ):

            if isinstance(curr_images[0], str):
                curr_images = [PIL.Image.open(i) for i in curr_images]

            dataset = Dataset.from_dict({"text": curr_prompts, "image": curr_images})
            dataset = dataset.map(
                lambda x: self.processor(x["text"], x["image"], **processor_args),
                batched=True,
                remove_columns=["text", "image"],
                desc="Processing macro batch",
            )

            collator = DataCollatorWithPadding(
                self.processor.tokenizer, pad_to_multiple_of=8, return_tensors="pt"
            )

            def base_loop(batch_size):
                """Base loop for generation."""

                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=collator,
                    # sampler=DistributedEvalSampler(dataset) if self.is_ddp else None,
                    pin_memory=True,
                )

                # if show_progress_bar is None:
                #     show_progress_bar = True if len(texts) > 1 else False

                outputs = list()
                for batch_idx, batch in tqdm(
                    enumerate(loader),
                    desc="Generation",
                    total=len(loader),
                    disable=not show_progress_bar or self.local_rank != 0,
                ):
                    batch = batch.to(self.model.device)

                    output = self.model.generate(**batch, **generation_kwargs)

                    # if self.vlm_type == VLMType.LLAVA:
                    #     inputs = self.processor(
                    #         batch["text"],
                    #         image,
                    #         return_tensors="pt",
                    #         # apply_chat_template=apply_chat_template,
                    #         # add_generation_prompt=add_generation_prompt,
                    #     ).to(self.model.device)
                    #     # current_generation_args = self._prepare_generation_args(**generation_kwargs)
                    #     output = self.model.generate(**inputs, **generation_kwargs)

                    # elif self.vlm_type == VLMType.IDEFICS:
                    #     prompt = IdeficsHelper.apply_chat_template(image, text)
                    #     inputs = self.processor(
                    #         prompt, add_end_of_utterance_token=False, return_tensors="pt"
                    #     ).to(self.model.device)
                    #     exit_condition = self.processor.tokenizer(
                    #         "<end_of_utterance>", add_special_tokens=False
                    #     ).input_ids
                    #     bad_words_ids = self.processor.tokenizer(
                    #         ["<image>", "<fake_token_around_image>"], add_special_tokens=False
                    #     ).input_ids
                    #     output = self.model.generate(
                    #         **inputs,
                    #         eos_token_id=exit_condition,
                    #         bad_words_ids=bad_words_ids,
                    #         **generation_kwargs,
                    #     )

                    # elif self.vlm_type == VLMType.BLIP2:
                    #     inputs = self.processor(image, text, return_tensors="pt").to(
                    #         self.model.device
                    #     )
                    #     output = self.model.generate(**inputs, **generation_kwargs)

                    # else:
                    #     RuntimeError()

                    if skip_prompt:
                        output = output[:, len(batch["input_ids"][0]) :]

                    decoded = self.processor.decode(output[0], skip_special_tokens=True)
                    outputs.append(decoded)

                return outputs

            @find_executable_batch_size(starting_batch_size=starting_batch_size)
            def find_batch_size_loop(batch_size):
                logger.info(f"Auto finding batch size... Testing bs={batch_size}")
                return base_loop(batch_size)

            if batch_size == "auto":
                logger.info(
                    f"Finding the optimal batch size... Starting with {starting_batch_size}"
                )
                macro_batch_responses = find_batch_size_loop()
            else:
                macro_batch_responses = base_loop(batch_size)

            responses.append(macro_batch_responses)

        return responses

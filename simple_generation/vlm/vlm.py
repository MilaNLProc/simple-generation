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
from ..utils import DistributedEvalSampler


logger = get_logger(__name__)

inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


class VLMType(Enum):
    LLAVA = "LLAVA"
    IDEFICS = "IDEFICS"
    QWEN = "QWEN"
    BLIP2 = "BLIP2"


@dataclasses.dataclass
class DefaultGenerationConfig(GenerationConfig):
    """Default generation configuration.

    We apply this parameters to any .generate() call, unless they are not overridden.

    Attributes:
        max_new_tokens (int): The maximum number of tokens to generate. Defaults to 512.
        do_sample (bool): Whether to use sampling or greedy decoding. Defaults to True.
        temperature (float): The sampling temperature. Defaults to 0.7.
        top_p (float): The cumulative probability for sampling from the top_p distribution. Defaults to 1.0.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to 50.
        num_return_sequences (int): The number of independently computed returned sequences for each element in the batch. Defaults to 1.

    """

    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    num_return_sequences: int = 1


class SimpleVLMGenerator:
    @property
    def is_ddp(self):
        """Returns True if the model is distributed."""
        return dist.is_available() and dist.is_initialized()

    @property
    def local_rank(self):
        """Returns the local rank of the process. If not in DDP, returns 0."""
        return dist.get_rank() if self.is_ddp else 0

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def __init__(self, model_name_or_path, **model_kwargs):
        self.model_name_or_path = model_name_or_path

        # Use accelerator to distribute model if DDP is enabled
        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        user_request_move_to_device = False

        if "device" in model_kwargs:
            logger.info(f"Setting device to {self.device} per user's request.")
            self.device = model_kwargs.pop("device")
            user_request_move_to_device = True

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
        self.tokenizer.padding_side = "left"

        try:
            self.generation_config = GenerationConfig.from_pretrained(
                model_name_or_path
            )
        except Exception as e:
            logger.warning("Could not load generation config. Using default one.")
            self.generation_config = DefaultGenerationConfig()

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

    def _prepare_generation_args(self, **generation_kwargs):
        current_generation_args = self.generation_config.to_dict()

        logger.info("Setting pad_token_id to eos_token_id for open-end generation")
        current_generation_args["pad_token_id"] = self.tokenizer.eos_token_id
        current_generation_args["eos_token_id"] = self.tokenizer.eos_token_id

        # We fix when some model default to the outdated "max_length" parameter
        if "max_length" in current_generation_args:
            logger.info(
                "Found 'max_length' in the model's default generation config. Setting this value to 'max_new_tokens' instead."
            )
            current_generation_args["max_new_tokens"] = current_generation_args.pop(
                "max_length"
            )

        if len(generation_kwargs) > 0:
            logger.info(
                "Custom generation args passed. Any named parameters will override the same default one."
            )
            current_generation_args.update(generation_kwargs)

        # Postprocess generation kwargs
        if (
            "temperature" in current_generation_args
            and current_generation_args["temperature"] == 0
        ):
            logger.info("Temperature cannot be 0. Setting it to 1e-4.")
            current_generation_args["temperature"] = 1e-4

        return current_generation_args

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
        log_batch_sample=-1,
        show_progress_bar=None,
        macro_batch_size: int = 512,
        **generation_kwargs,
    ):
        """
        TBD.
        """

        if not isinstance(texts, list):
            logger.debug("Texts is not a list. Wrapping it in a list.")
            texts = [texts]
        if not isinstance(images, list):
            logger.debug("Images is not a list. Wrapping it in a list.")
            images = [images]

        if len(texts) != len(images):
            raise ValueError("Prompt and image counts must be the same.")

        if show_progress_bar is None:
            show_progress_bar = True if len(texts) > 1 else False

        current_generation_args = self._prepare_generation_args(**generation_kwargs)
        logger.debug("Generation args:", current_generation_args)

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

        batch_starts = range(0, len(texts), macro_batch_size)
        responses = list()
        for batch_start_id in tqdm(
            batch_starts,
            desc="Macro batch",
            total=math.ceil(len(texts) / macro_batch_size),
            disable=(len(texts) <= macro_batch_size),
        ):
            curr_prompts = texts[batch_start_id : batch_start_id + macro_batch_size]
            curr_images = images[batch_start_id:macro_batch_size]

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
                    sampler=DistributedEvalSampler(dataset) if self.is_ddp else None,
                    pin_memory=True,
                )

                outputs = list()
                for batch_idx, batch in tqdm(
                    enumerate(loader),
                    desc="Generation",
                    total=len(loader),
                    disable=not show_progress_bar or self.local_rank != 0,
                ):
                    batch = batch.to(self.model.device)
                    try:
                        output = self.model.generate(**batch, **current_generation_args)

                        if skip_prompt:
                            output = output[:, len(batch["input_ids"][0]) :]

                        decoded = self.processor.batch_decode(
                            output, skip_special_tokens=True
                        )
                    except Exception as e:
                        if isinstance(e, torch.cuda.OutOfMemoryError):
                            raise e

                        logger.error(f"Error {e}")
                        logger.error("Generation failed. Skipping batch.")
                        decoded = ["ERROR: Generation failed"] * len(batch["input_ids"])

                    outputs.extend(decoded)

                    if log_batch_sample != -1 and (
                        log_batch_sample % (batch_idx + 1) == 0
                    ):
                        logger.info(
                            f"Log decoded text at batch_id {batch_idx}", decoded[0]
                        )

                if self.is_ddp:
                    target_list = [None for _ in range(dist.get_world_size())]

                    dist.gather_object(
                        outputs, target_list if dist.get_rank() == 0 else None, dst=0
                    )

                    if self.is_main_process:
                        responses = [
                            item for sublist in target_list for item in sublist
                        ]
                    else:
                        logger.debug(
                            f"Killing non-main process with rank {dist.get_rank()} as no longer needed."
                        )
                        exit(0)

                else:
                    responses = outputs

                return responses

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

            responses.extend(macro_batch_responses)

        return responses

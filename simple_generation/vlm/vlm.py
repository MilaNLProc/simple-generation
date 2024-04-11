import PIL.Image
import torch
import torch.distributed as dist

from transformers import (
    GenerationConfig,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import find_executable_batch_size
from codecarbon import track_emissions
import PIL
import dataclasses
from typing import List, Union
from tqdm import tqdm

logger = get_logger(__name__)

inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


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

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        try:
            self.generation_config = GenerationConfig.from_pretrained(
                model_name_or_path
            )
        except Exception as e:
            logger.warning("Could not load generation config. Using default one.")
            self.generation_config = DefaultGenerationConfig()

        if "llava" in self.model_name_or_path.lower():
            model_cls = LlavaNextForConditionalGeneration
        else:
            raise NotImplementedError(f"We do not wrap {model_name_or_path} yet!")

        self.model = model_cls.from_pretrained(model_name_or_path, **model_kwargs)

        if self.is_ddp or user_request_move_to_device:
            self.model.to(self.device)
            logger.debug(f"Moving model to {self.device}")

        self.model.eval()

        print(
            f"""
            Simple Generation (VLM) initialization completed!

            Model placement:
            - device_map: {model_kwargs.pop('device_map', None)},
            - device: {self.device},

            DDP:
            - distributed inference: {self.is_ddp},
            """
        )

    @track_emissions(log_level="error", measure_power_secs=60)
    @inference_decorator()
    def __call__(
        self,
        texts,
        images,
        # batch_size="auto",
        # starting_batch_size=256,
        # num_workers=4,
        # skip_prompt=False,
        # log_batch_sample=-1,
        show_progress_bar=None,
        # prepare_prompts=False,  # keeping it here for consistency
        # apply_chat_template=False,
        # add_generation_prompt=False,
        **generation_kwargs,
    ):
        """
        v1:
        - no support to batched inference
        - the user needs to take care of the input format
        """

        if not isinstance(texts, list):
            logger.debug("Texts is not a list. Wrapping it in a list.")
            texts = [texts]
        if not isinstance(images, list):
            logger.debug("Images is not a list. Wrapping it in a list.")
            images = [images]

        if len(texts) != len(images):
            raise ValueError("Prompt and image counts must be the same.")

        if isinstance(images[0], str):
            logger.info(
                "Checking the first image we found a string. Trying to load the list from disk... (Note that large image collections might saturate your RAM)"
            )
            images = [PIL.Image.open(i) for i in images]

        if show_progress_bar is None:
            show_progress_bar = True if len(texts) > 1 else False

        outputs = list()
        for text, image in tqdm(
            zip(texts, images),
            desc="Item",
            total=len(texts),
            disable=not show_progress_bar,
        ):
            inputs = self.processor(
                text,
                image,
                return_tensors="pt",
                # apply_chat_template=apply_chat_template,
                # add_generation_prompt=add_generation_prompt,
            )
            inputs = inputs.to(self.model.device)

            # current_generation_args = self._prepare_generation_args(**generation_kwargs)

            output = self.model.generate(**inputs, **generation_kwargs)

            decoded = self.processor.decode(output[0], skip_special_tokens=True)
            outputs.append(decoded)

        return outputs

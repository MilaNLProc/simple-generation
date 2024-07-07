"""Main module."""

import dataclasses
from typing import List, Dict

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import find_executable_batch_size
from codecarbon import track_emissions
from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
)
import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread

from .utils import DistributedEvalSampler

logger = get_logger(__name__)


inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


class SimpleGenerator:
    """
    SimpleGenerator is a wrapper around Hugging Face's Transformers library that allows for easy generation of text from a given prompt.
    """

    @property
    def local_rank(self):
        """Returns the local rank of the process. If not in DDP, returns 0."""
        return dist.get_rank() if self.is_ddp else 0

    @property
    def is_ddp(self):
        """Returns True if the model is distributed."""
        return dist.is_available() and dist.is_initialized()

    @property
    def is_main_process(self):
        """Returns True if the process is the main process."""
        return self.accelerator.is_main_process

    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path=None,
        lora_weights=None,
        compile_model=False,
        use_bettertransformer=False,
        **model_kwargs,
    ):
        """Initialize the SimpleGenerator.

        Args:
            model_name_or_path (str): The model name or path to load from.
            tokenizer_name_or_path (str, optional): The tokenizer name or path to load from. Defaults to None, in which case it will be set to the model_name_or_path.
            lora_weights (str, optional): The path to the LoRA weights. Defaults to None.
            compile_model (bool, optional): Whether to torch.compile() the model. Defaults to False.
            use_bettertransformer (bool, optional): Whether to transform the model with BetterTransformers. Defaults to False.
            **model_kwargs: Any other keyword arguments will be passed to the model's from_pretrained() method.

        Returns:
            SimpleGenerator: The SimpleGenerator object.

        Examples:
            >>> from simple_generation import SimpleGenerator
            >>> generator = SimpleGenerator("meta-llama/Llama-2-7b-chat-hf", apply_chat_template=True)
        """
        self.model_name_or_path = model_name_or_path

        # Use accelerator to distribute model if DDP is enabled
        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        user_request_move_to_device = False

        if "device" in model_kwargs:
            logger.info(f"Setting device to {self.device} per user's request.")
            self.device = model_kwargs.pop("device")
            user_request_move_to_device = True

        # Load config and inspect whether the model is a seq2seq or causal LM
        config = None
        trust_remote_code = model_kwargs.get("trust_remote_code", False)
        try:
            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code
            )

            if config.architectures == "LLaMAForCausalLM":
                logger.warning(
                    "We found a deprecated LLaMAForCausalLM architecture in the model's config and updated it to LlamaForCausalLM."
                )
                config.architectures == "LlamaForCausalLM"

            is_encoder_decoder = getattr(config, "is_encoder_decoder", None)
            if is_encoder_decoder == None:
                logger.warning(
                    "Could not find 'is_encoder_decoder' in the model config. Assuming it's an autoregressive model."
                )
                is_encoder_decoder = False

            model_kwargs["config"] = config

        except:
            logger.warning(
                f"Could not find config in {model_name_or_path}. Assuming it's an autoregressive model."
            )
            is_encoder_decoder = False

        self.is_encoder_decoder = is_encoder_decoder

        if is_encoder_decoder:
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModelForCausalLM

        tokenizer_name = (
            tokenizer_name_or_path if tokenizer_name_or_path else model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, config=config, padding_side="left"
        )

        # padding_size="left" is required for autoregressive models, and should not make a difference for every other model as we use attention_masks. See: https://github.com/huggingface/transformers/issues/3021#issuecomment-1454266627 for a discussion on why left padding is needed on batched inference
        # This is also relevant for VLM batched generation: https://huggingface.co/docs/transformers/model_doc/llava_next#usage-tips
        self.tokenizer.padding_side = "left"

        logger.debug("Setting off the deprecation warning for padding")
        # see https://github.com/huggingface/transformers/issues/22638
        # and monitor https://github.com/huggingface/transformers/pull/23742
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        if not getattr(self.tokenizer, "pad_token", None):
            logger.warning(
                "Couldn't find a PAD token in the tokenizer, using the EOS token instead."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        if lora_weights:
            logger.info("Attaching LoRA weights to the model")
            self.model = PeftModel.from_pretrained(self.model, lora_weights)

        if use_bettertransformer:
            logger.info("Transforming model with bettertransformer")
            try:
                from optimum.bettertransformer import BetterTransformer

                self.model = BetterTransformer.transform(self.model)
            except Exception as e:
                print(e)
                logger.error("Couldn't transformer the model with BetterTransformers")

        if compile_model:
            logger.info("torch.compiling the model")
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(e)
                logger.error(
                    "Couldn't torch.compile the model. Check that your torch version is >=2.*"
                )

        self.model.eval()

        print(
            f"""
            Simple Generation initialization completed!

            Model placement:
            - device_map: {model_kwargs.pop('device_map', None)},
            - device: {self.device},

            DDP:
            - distributed inference: {self.is_ddp},

            Model info:
            - is_encoder_decoder: {self.is_encoder_decoder},
            - lora_weights: {lora_weights},
            - use_bettertransformer: {use_bettertransformer},
            - compile_model: {compile_model}
            """
        )

    def conversation_from_user_prompts(
        self,
        user_prompts: List[str],
        **kwargs,
    ) -> List[Dict]:
        """Generate a multi-turn conversation with multiple user prompts.

        Generate a conversation out of several user prompts. I.e., every user prompt is fed to the model and the response is appended to the history. The history is then fed to the model again, and so on.
        Note that this operation is not batched.

        Args:
            user_prompts (List[str]): A list of turn texts. Each element is the human written text for a turn.
            return_last_response (bool, optional): If True, the last response is returned as well. Defaults to False.

        Returns:
            List[Dict]: A list containing the conversation, one item per turn, following the Hugging Face chat template format.
        """

        conversation = list()
        for user_prompt in tqdm(user_prompts, desc="Turns"):
            conversation.append({"role": "user", "content": user_prompt})
            conv_text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            response = self(
                conv_text,
                skip_prompt=True,
                show_progress_bar=False,
                apply_chat_template=False,
                **kwargs,
            )

            # append the model's response to the conversation
            conversation.append({"role": "assistant", "content": response[0]})

        return conversation

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
        batch_size="auto",
        starting_batch_size=256,
        num_workers=4,
        skip_prompt=False,
        log_batch_sample=-1,
        show_progress_bar=True,
        prepare_prompts=False,  # keeping it here for consistency
        apply_chat_template=False,
        add_generation_prompt=False,
        **generation_kwargs,
    ):
        """Generate text from a given prompt.

        Args:
            texts (str or List[str]): The text prompt(s) to generate from.
            batch_size (int, optional): The batch size to use for generation. Defaults to "auto", in which case it will be found automatically.
            starting_batch_size (int, optional): The starting batch size to use for finding the optimal batch size. Defaults to 256.
            num_workers (int, optional): The number of workers to use for the DataLoader. Defaults to 4.
            skip_prompt (bool, optional): Whether to skip the initial prompt when returning the generated text. Defaults to False. Set it to False if you are using a sequence to sequence model.
            log_batch_sample (int, optional): If >0, every log_batch_sample batches the output text will be logged. Defaults to -1.
            show_progress_bar (bool, optional): Whether to show the progress bar. Defaults to True.
            apply_chat_template (bool, optional): Whether to apply the chat template to the prompts. Defaults to False.
            add_generation_prompt (bool, optional): Whether to add the generation prompt to the prompts. Defaults to False.
            **generation_kwargs: Any other keyword arguments will be passed to the model's generate() method.

        Returns:
            str or List[str]: The generated text(s).

        Examples:
            >>> from simple_generation import SimpleGenerator
            >>> generator = SimpleGenerator("meta-llama/Llama-2-7b-chat-hf", apply_chat_template=True)
            >>> generator("Tell me what's 2 + 2.", max_new_tokens=16, do_sample=True, top_k=50, skip_prompt=True)
            "The answer is 4."
        """
        # make texts a list if it's not
        if not isinstance(texts, list):
            logger.debug("Texts is not a list. Wrapping it in a list.")
            texts = [texts]

        if prepare_prompts:
            raise ValueError(
                "The argument 'prepare_prompts' has been deprecated. Set 'apply_chat_template=True' instead."
            )
            texts = self.prepare_prompts(texts)

        if apply_chat_template:
            texts = self._apply_chat_template_user(texts, add_generation_prompt)

        current_generation_args = self._prepare_generation_args(**generation_kwargs)
        logger.debug("Generation args:", current_generation_args)

        # Processing the input text
        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=True),
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing texts",
        )

        collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt"
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
                    output = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **current_generation_args,
                    )

                    # remove initial text prompt from responses
                    if skip_prompt:
                        output = output[:, len(batch["input_ids"][0]) :]

                    decoded = self.tokenizer.batch_decode(
                        output, skip_special_tokens=True
                    )

                except Exception as e:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        raise e

                    logger.error(f"Error {e}")
                    logger.error("Generation failed. Skipping batch.")
                    decoded = ["ERROR: Generation failed"] * len(batch["input_ids"])

                outputs.extend(decoded)

                if log_batch_sample != -1 and (log_batch_sample % (batch_idx + 1) == 0):
                    logger.info(f"Log decoded text at batch_id {batch_idx}", decoded[0])

            if self.is_ddp:
                target_list = [None for _ in range(dist.get_world_size())]

                dist.gather_object(
                    outputs, target_list if dist.get_rank() == 0 else None, dst=0
                )

                if self.is_main_process:
                    responses = [item for sublist in target_list for item in sublist]
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
            responses = find_batch_size_loop()
        else:
            responses = base_loop(batch_size)

        return responses

    def _apply_chat_template_user(self, texts, add_generation_prompt):
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            for t in texts
        ]

    def gui(self, type: str = "chat", **generation_kwargs):
        """(Deprecated) Start a GUI for the model."""
        raise DeprecationWarning(
            "GUI cannot be launched from this class anymore. Use the CLI as indicated in the README."
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
    # num_beams: int = 1
    # early_stopping: bool = False
    # repetition_penalty: float = 1.0
    # typical_p: float = 1.0
    # penalty_alpha: float = 0.2
    # length_penalty: int = 1.2

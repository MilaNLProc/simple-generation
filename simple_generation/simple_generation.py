"""Main module."""
import dataclasses
from typing import List

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

from .conversation import PromptHandler
from .utils import DistributedEvalSampler

logger = get_logger(__name__)


inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


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
    # num_beams: int = 1
    # early_stopping: bool = False
    # repetition_penalty: float = 1.0
    # typical_p: float = 1.0
    # penalty_alpha: float = 0.2
    # length_penalty: int = 1.2


class SimpleGenerator:
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path=None,
        lora_weights=None,
        compile_model=False,
        use_bettertransformer=False,
        system_prompt=None,
        system_message=None,
        **model_kwargs,
    ):
        # Use accelerator to distribute model if DDP is enabled
        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        self.is_ddp = True if dist.is_initialized() else False

        self.system_prompt = system_prompt
        self.system_message = system_message
        if self.system_prompt is None:
            logger.warning(
                "No system prompt template specified. Be aware that your generation will use your plain input."
            )

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

        # By default we use HF's smart device placement strategy for model weights
        if "device_map" not in model_kwargs and not self.is_ddp:
            logger.info("Setting the device map to 'auto' since not specified")
            model_kwargs["device_map"] = "auto"
        try:
            self.model = model_cls.from_pretrained(model_name_or_path, **model_kwargs)

            if self.is_ddp:
                self.model.to(self.device)
                logger.debug(f"Sending model to {self.device}")
        except:
            model_kwargs.pop("device_map")
            logger.debug("Removig device_map and trying loading model again")
            self.model = model_cls.from_pretrained(model_name_or_path, **model_kwargs)

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

    @property
    def local_rank(self):
        return dist.get_rank() if self.is_ddp else 0

    def conversation_from_user_prompts(
        self,
        user_prompts: List[str],
        return_conversation: bool = True,
        return_last_response: bool = False,
        **kwargs,
    ):
        """Generate a multi-turn conversation with multiple user prompts.

        Generate a conversation out of several user prompts. I.e., every user prompt is fed to the model and the response is appended to the history. The history is then fed to the model again, and so on.
        Note that this operation is not batched.

        Args:
            user_prompts (List[str]): A list of turn texts. Each element is the human written text for a turn.
            return_last_response (bool, optional): If True, the last response is returned as well. Defaults to False.

        Returns:
            str: The generated conversation.
        """

        if not self.system_prompt:
            raise NotImplementedError(
                "No system prompt was provided to the constructor."
            )

        ph = PromptHandler(self.system_prompt)

        for user_prompt in tqdm(user_prompts, desc="Turns"):
            ph.append_message("user", user_prompt)
            ph.append_message("system", None)
            query = ph.build_prompt()

            response = self(
                query,
                skip_prompt=True,
                show_progress_bar=False,
                prepare_prompts=False,
                **kwargs,
            )
            response = response[0]

            ph.conversation.messages = ph.conversation.messages[:-1]
            ph.append_message("system", response)

        output = ()
        if return_conversation:
            conversation = ph.build_prompt()
            output = (conversation,)

        if return_last_response:
            output += (response,)

        return output

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
        prepare_prompts=True,
        **generation_kwargs,
    ):
        # make texts a list if it's not
        if not isinstance(texts, list):
            logger.debug("Texts is not a list. Wrapping it in a list.")
            texts = [texts]

        if prepare_prompts:
            texts = self.prepare_prompts(texts)

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

        logger.debug("Generation args:", current_generation_args)

        # Processing the input text
        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(
            lambda x: self.tokenizer(x["text"]), batched=True, remove_columns=["text"]
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
                sampler=DistributedEvalSampler(dataset),
                pin_memory=True,
            )

            # if self.is_ddp:
            #     loader = self.accelerator.prepare(loader)

            print("Loader len:", len(loader))

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

                # if self.is_ddp:
                #     decoded = [
                #         (dist.get_rank(), batch_idx, idx, text)
                #         for idx, text in enumerate(decoded)
                #     ]

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
                    exit(0)  # we do not need the process anymore

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

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    def prepare_prompts(self, texts):
        """
        Prepare the prompts for generation.
        """

        # Set a conversation template that will be applied to every prompt generation
        if self.system_prompt is not None:
            logger.info(
                f"Using system prompt associate with the model: {self.system_prompt}. "
                "See https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py for more details."
            )

            if self.system_message is not None:
                logger.info(
                    f"Using system message associate with the model: {self.system_message}."
                )

            new_texts = list()
            for text in texts:
                ph = PromptHandler(self.system_prompt, self.system_message)

                ph.append_message("user", text)
                ph.append_message("system", None)
                new_texts.append(ph.build_prompt())

            texts = new_texts

        return texts

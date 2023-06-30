"""Main module."""
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    GenerationConfig,
    AutoConfig,
)
from tqdm import tqdm
from datasets import Dataset
import torch
from codecarbon import track_emissions
import dataclasses
from peft import PeftModel
from accelerate.utils import find_executable_batch_size


import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DefaultGenerationConfig(GenerationConfig):

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


class SimpleGenerator:
    def __init__(
        self,
        model_name_or_path,
        lora_weights=None,
        tokenizer_name_or_path=None,
        device_map="auto",
        load_in_8bit=False,
        load_in_4bit=False,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path)
        is_encoder_decoder = getattr(config, "is_encoder_decoder", None)
        if is_encoder_decoder == None:
            logger.warning(
                "Could not find 'is_encoder_decoder' in the model config. Assuming it's a seq2seq model."
            )
            is_encoder_decoder = False

        if is_encoder_decoder:
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModelForCausalLM

        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot load in both 4bit and 8bit")

        tokenizer_name = (
            tokenizer_name_or_path if tokenizer_name_or_path else model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, padding_side="left"
        )

        if not getattr(self.tokenizer, "pad_token", None):
            logger.warning(
                "Couldn't find a PAD token in the tokenizer, using the EOS token instead."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_args = {
            "device_map": device_map,
            "load_in_8bit": load_in_8bit,
            "load_in_4bit": load_in_4bit,
        }

        try:
            self.generation_config = GenerationConfig.from_pretrained(
                model_name_or_path
            )
        except Exception as e:
            logger.warning("Could not load generation config. Using default one.")
            self.generation_config = DefaultGenerationConfig()

        self.model = model_cls.from_pretrained(model_name_or_path, **model_args)

        if lora_weights:
            logger.info("Attaching LoRA weights to the model")
            self.model = PeftModel.from_pretrained(self.model, lora_weights)

        self.model.eval()

    @track_emissions
    @torch.no_grad()
    def __call__(
        self,
        texts,
        batch_size="auto",
        prefix=None,
        num_workers=4,
        return_full_text=True,
        **generation_kwargs,
    ):
        if not isinstance(texts, list):
            logger.debug("Texts is not a list. Wrapping it in a list.")
            texts = [texts]

        if prefix:
            logger.info("Prefix is set. Adding it to each text.")
            texts = [f"{prefix}{text}" for text in texts]

        current_generation_args = self.generation_config.to_dict()

        logger.info("Setting pad_token_id to eos_token_id for open-end generation")
        current_generation_args["pad_token_id"] = self.tokenizer.eos_token_id
        current_generation_args["eos_token_id"] = self.tokenizer.eos_token_id

        logger.info("Using the new 'max_new_tokens' parameter")
        current_generation_args["max_new_tokens"] = current_generation_args.pop(
            "max_length", 20
        )

        if len(generation_kwargs) > 0:
            logger.info(
                "Custom generation args passed. Any named parameters will override the same default one."
            )
            current_generation_args.update(generation_kwargs)

        logger.info("Generation args:", current_generation_args)

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
                pin_memory=True,
            )

            output_texts = list()
            for batch in tqdm(loader, desc="Generation"):
                batch = batch.to(self.model.device)
                try:
                    output = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **current_generation_args,
                    )
                    decoded = self.tokenizer.batch_decode(
                        output, skip_special_tokens=True
                    )

                except Exception as e:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        raise e

                    logger.error("Error", e)
                    logger.error("Generation failed. Skipping batch.")
                    decoded = [""] * len(batch["input_ids"])
                output_texts.extend(decoded)

            return output_texts

        @find_executable_batch_size(starting_batch_size=128)
        def find_batch_size_loop(batch_size):
            return base_loop(batch_size)

        if batch_size == "auto":
            logger.info(f"Finding the optimal batch size... Starting with 128")
            responses = find_batch_size_loop()
        else:
            responses = base_loop(batch_size)

        # remove initial text prompt form responses
        if not return_full_text:
            responses = [r.split(texts[i])[-1] for i, r in enumerate(responses)]

        return responses

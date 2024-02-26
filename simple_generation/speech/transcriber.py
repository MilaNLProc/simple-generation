from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from .config import SEAMLESS_CODE_TO_LANG, WHISPER_CODE_TO_LANG


logger = get_logger(__name__)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, array_list):
        self.array_list = array_list

    def __getitem__(self, index):
        return self.array_list[index]

    def __len__(self):
        return len(self.array_list)


class SimpleTranscriber:
    def __init__(
        self,
        model_name_or_path: str,
        tgt_lang: str,
        device: Optional[str] = None,
        **init_kwargs,
    ):
        self.tgt_lang = tgt_lang
        self.model_name_or_path = model_name_or_path

        self.accelerator = Accelerator(device_placement=True)
        self.device = self.accelerator.device
        if device:
            logger.info("Overriding device as per user request to:", device)
            self.device = device

        # # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Load Model
        if "whisper" in model_name_or_path or "seamless" in model_name_or_path:
            model_cls = AutoModelForSpeechSeq2Seq
        else:
            model_cls = AutoModelForCTC
            raise ValueError("Only whisper and seamless models are supported")

        logger.info(f"Loading {model_name_or_path} and moving it to {device}...")
        self.model = (
            model_cls.from_pretrained(model_name_or_path, **init_kwargs)
            .to(device)
            .eval()
        )

        logger.info("Transcriber loaded for language:", tgt_lang)

    def _build_loader(
        self,
        raw_audio: List[Union[np.ndarray, List[float]]],
        sampling_rate: int,
        batch_size: int,
        num_workers: int,
        max_length: int,
    ):
        pargs = dict(
            return_tensors="pt",
            sampling_rate=sampling_rate,
            return_attention_mask=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_length,
        )

        def collate_pad_and_trim(batch: List[Union[np.ndarray, List[float]]]):
            """
            Pad/trim all audios to a max length. Then, create a batch.
            """
            if "whisper" in self.model_name_or_path:
                pargs["audio"] = batch  # type: ignore
                pargs["do_normalize"] = True
                pargs["padding"] = "max_length"
            else:
                pargs["audios"] = batch  # type: ignore
                pargs["padding"] = "longest"

            inputs = self.processor(**pargs)
            return inputs

        loader = torch.utils.data.DataLoader(
            SimpleDataset(raw_audio),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_pad_and_trim,
            pin_memory=True,
        )

        return loader

    @torch.inference_mode()
    def __call__(
        self,
        raw_audio: List[Union[np.ndarray, List[float]]],
        sampling_rate: int,
        batch_size: int = 1,
        num_workers: int = 1,
        show_progress_bar: bool = True,
        max_length: int = 480000,
        **generation_kwargs,
    ):
        """
        Transcribe a list of audio samples.

        Args:
            raw_audio (List[Union[np.ndarray, List[float]]]): List of raw audio data.
            sampling_rate (int): Sampling rate of the audio data.
            batch_size (int, optional): Number of audio samples per batch. Defaults to 1.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 1.
            show_progress_bar (bool, optional): Whether to show a progress bar during inference. Defaults to True.
            max_length (int, optional): Maximum length of audio samples in the batch. Defaults to 480000.
            **generation_kwargs: Additional keyword arguments for the generation process.

        Returns:
            List[str]: List of transcriptions for each audio sample.
        """

        loader = self._build_loader(
            raw_audio=raw_audio,
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
        )

        transcriptions = list()
        for idx, batch in tqdm(
            enumerate(loader),
            desc="Batch",
            disable=not show_progress_bar,
            total=len(loader),
        ):
            batch = {
                k: v.to(dtype=self.model.dtype, device=self.model.device)
                for k, v in batch.items()
            }

            if "whisper" in self.model_name_or_path:
                generation_kwargs["forced_decoder_ids"] = (
                    self.processor.get_decoder_prompt_ids(
                        language=WHISPER_CODE_TO_LANG[self.tgt_lang], task="transcribe"
                    )
                )

                predicted_ids = self.model.generate(**batch, **generation_kwargs)

            elif "seamless" in self.model_name_or_path:
                generation_kwargs |= {
                    "tgt_lang": SEAMLESS_CODE_TO_LANG[self.tgt_lang],
                }

                predicted_ids = self.model.generate(**batch, **generation_kwargs)
            else:  # A CTC model
                raise ValueError("Only whisper and seamless models are supported")
                logits = self.model(**batch, **generation_kwargs).logits
                predicted_ids = logits.argmax(-1)

            results = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            transcriptions.extend(results)

        return transcriptions

    def forward_encoder(
        self,
        raw_audio: List[Union[np.ndarray, List[float]]],
        sampling_rate: int,
        batch_size: int = 1,
        num_workers: int = 1,
        show_progress_bar: bool = False,
        max_length: int = 480000,
        **forward_kwargs,
    ):
        output_hidden_states = forward_kwargs.get("output_hidden_states", False)

        loader = self._build_loader(
            raw_audio=raw_audio,
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
        )

        hidden_states_list = list()

        for batch in tqdm(loader, desc="Batch", disable=not show_progress_bar):

            batch = batch.to(dtype=self.model.dtype, device=self.device)

            out = self.model.model.encoder(**batch, **forward_kwargs)

            if output_hidden_states:
                last_hs = out.hidden_states[-1]  # (bs, seq_len, hsize)
                hidden_states_list.append(last_hs.cpu().detach())

        output = dict()

        if output_hidden_states:
            # (dataset, num_layers, seq_len, hs)
            output["encoder_hidden_states"] = torch.cat(hidden_states_list)

        return output

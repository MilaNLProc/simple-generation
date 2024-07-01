from dataclasses import dataclass
from transformers import AutoProcessor
from typing import Mapping
from enum import Enum
import pdb


class VLMType(Enum):
    LLAVA = "LLAVA"
    LLAVA_NEXT = "LLAVA_NEXT"
    IDEFICS = "IDEFICS"
    IDEFICS2 = "IDEFICS2"
    BLIP2 = "BLIP2"


@dataclass
class VLMCollator:
    vlm_type: VLMType
    processor: AutoProcessor
    processor_args: Mapping

    def __call__(self, batch):
        prompts = [x["text"] for x in batch]
        images = [x["image"] for x in batch] if "image" in batch else None

        if self.vlm_type == VLMType.LLAVA:
            if images:
                prompts = [f"USER: <image>\n{p}\nASSISTANT:" for p in prompts]
                batch = self.processor(prompts, images, **self.processor_args)
            else:
                prompts = [f"USER: {p}\nASSISTANT:" for p in prompts]
                batch = self.processor(text=prompts, **self.processor_args)
                batch.pop("pixel_values")

        elif self.vlm_type == VLMType.LLAVA_NEXT:

            if images:
                prompts = [f"[INST] <image>\n{p} [/INST]" for p in prompts]
                batch = self.processor(prompts, images, **self.processor_args)
            else:
                prompts = [f"[INST] {p} [/INST]" for p in prompts]
                batch = self.processor(prompts, **self.processor_args)

        elif self.vlm_type == VLMType.IDEFICS:

            if images:
                inputs = [
                    [f"User: {p}", i, "<end_of_utterance>", "\nAssistant:"]
                    for p, i in zip(prompts, images)
                ]
            else:
                inputs = [
                    [f"User: {p}", "<end_of_utterance>", "\nAssistant:"]
                    for p in prompts
                ]

            batch = self.processor(inputs, **self.processor_args)

        elif self.vlm_type == VLMType.IDEFICS2:

            if images:
                messages = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": p},
                            ],
                        },
                    ]
                    for p in prompts
                ]
                prompts = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                batch = self.processor(
                    text=prompts, images=[[i] for i in images], **self.processor_args
                )
            else:
                messages = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": p},
                            ],
                        },
                    ]
                    for p in prompts
                ]
                prompts = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                batch = self.processor(text=prompts, **self.processor_args)

        elif self.vlm_type == VLMType.BLIP2:

            if images:
                batch = self.processor(images, prompts, **self.processor_args)
            else:
                batch = self.processor(text=prompts, **self.processor_args)

        else:
            raise RuntimeError("VLMType not supported")

        return batch

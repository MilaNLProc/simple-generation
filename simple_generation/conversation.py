from dataclasses import dataclass
from typing import List, Union, Iterable, Tuple

import fastchat.conversation as conversation

conv_templates = conversation.conv_templates
conv_templates["vicuna_one_shot"] = conv_templates.pop("one_shot")
conv_templates["vicuna_zero_shot"] = conv_templates.pop("zero_shot")


def available_system_prompts():
    return sorted(list(conv_templates.keys()))


class PromptHandler:
    def __init__(self, system_prompt) -> None:
        self.system_prompt = system_prompt
        self.conversation = conv_templates[system_prompt]
        self.conversation.messages = list()

    def append_message(self, role: str, message: str = None):
        role = (
            self.conversation.roles[0] if role == "user" else self.conversation.roles[1]
        )
        self.conversation.append_message(role, message)

    def build_prompt(self):
        """
        Build a prompt for the model to generate a response given the conversation history.
        """
        return self.conversation.get_prompt()

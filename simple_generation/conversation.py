from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

import fastchat.conversation as conversation

conv_templates = conversation.conv_templates
conv_templates["vicuna_one_shot"] = conv_templates.pop("one_shot")
conv_templates["vicuna_zero_shot"] = conv_templates.pop("zero_shot")


def available_system_prompts():
    return sorted(list(conv_templates.keys()))


class PromptHandler:
    def __init__(self, system_prompt: str, system_message: str = None) -> None:
        self.system_prompt = system_prompt
        self.system_message = system_message
        self.conversation = conv_templates[system_prompt]

        if system_message is not None:
            self.conversation.system = system_message

        self.conversation.messages = list()

    def append_message(self, role: str, message: str = None):
        role = (
            self.conversation.roles[0] if role == "user" else self.conversation.roles[1]
        )
        self.conversation.append_message(role, message)

    def update_last_message(self, message: str):
        self.conversation.update_last_message(message)

    def build_prompt(self):
        """
        Build a prompt for the model to generate a response given the conversation history.
        """
        return self.conversation.get_prompt()

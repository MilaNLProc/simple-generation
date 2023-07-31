from dataclasses import dataclass
from typing import List, Union, Iterable, Tuple

import fastchat.conversation as conversation

conv_templates = conversation.conv_templates
conv_templates["vicuna_one_shot"] = conv_templates.pop("one_shot")
conv_templates["vicuna_zero_shot"] = conv_templates.pop("zero_shot")


def available_system_prompts():
    return sorted(list(conv_templates.keys()))


class PromptHandler:
    def __init__(self, system_prompt: None) -> None:
        self.system_prompt = system_prompt
        if system_prompt is not None:
            self.conv_handler = ConversationHandler(system_prompt)

    def uses_system_prompt(self):
        return self.system_prompt is not None

    def build_prompt(self, message):
        if hasattr(self, "conv_handler"):
            self.conv_handler.add_turns([(message, None)])
            return self.conv_handler.build_propmt()
        else:
            return message


class ConversationHandler:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.conversation = conv_templates[model_name]
        self.conversation.messages = list()

    def add_turns(self, turns: List[Iterable], clean_history: bool = True):
        """
        Add turns to the conversation.

        Args:
            turns (List[Iterable]): A list of turns to be added to the conversation. Turns are iterables of length 2, where the first element is the message from the first role and the second element is the message from the second role. Typically, the first role is the user and the second role is the system.
            clean_history (bool, optional): If True, the conversation history is cleaned before adding the turns. Defaults to True.
        """
        self.conversation.messages = list()
        for turn in turns:
            self.conversation.append_message(self.conversation.roles[0], turn[0])
            self.conversation.append_message(self.conversation.roles[1], turn[1])

    def build_propmt(self):
        """
        Build a prompt for the model to generate a response given the conversation history.

        Returns:
            str: The prompt to be sent to the model.
        """
        return self.conversation.get_prompt()

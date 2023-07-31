from dataclasses import dataclass
from typing import List, Union

from fastchat.conversation import conv_templates


@dataclass
class ConversationHandler:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.conversation = conv_templates[model_name]

    def build_propmt(self, message: str):
        self.conversation.append_message(message)
        return self.conversation.get_prompt()

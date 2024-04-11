from typing import List


class IdeficsHelper:

    @classmethod
    def apply_chat_template(self, user_img, user_prompt):
        return [user_img, f"User: {user_prompt}", "<end_of_utterance>", "\nAssistant:"]

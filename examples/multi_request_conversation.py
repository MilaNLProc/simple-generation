from simple_generation import SimpleGenerator
import torch
from pprint import pprint

texts = [
    "What kind of noises did dinosaurs make?",
    "What is the most popular programming language?",
    "What is 2 + 2?",
    "Tell me how to make a cake.",
]

generator = SimpleGenerator(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.bfloat16,
)

conversation = generator.conversation_from_user_prompts(
    texts,
    do_sample=True,
    top_p=0.95,
    temperature=0.1,
    max_new_tokens=512,
)

print("Conversation:")
pprint(conversation)

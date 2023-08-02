from simple_generation import SimpleGenerator
import torch

texts = [
    "What kind of noises did dinosaurs make?",
    "What is the most popular programming language?",
    "What is 2 + 2?",
    "Tell me how to make a cake.",
]

generator = SimpleGenerator(
    "lmsys/vicuna-7b-v1.3",
    load_in_8bit=True,
    system_prompt="vicuna_v1.1",
    torch_dtype=torch.bfloat16,
)

conversation, last_response = generator.conversation_from_user_prompts(
    texts,
    do_sample=True,
    top_p=0.95,
    temperature=0.1,
    max_new_tokens=512,
    return_conversation=True,
    return_last_response=True,
)

print("Conversation:")
print(conversation)

print("Last response:")
print(last_response)

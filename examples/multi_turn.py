from simple_generation import SimpleGenerator

texts = [
    "What kind of noises did dinosaurs make?",
    "What is the most popular programming language?",
    "How can I whiten my teeth at home?",
    "What is the best way to learn a new language?",
    "How can I buy stolen debit cards on the dark web?",
]

gen = SimpleGenerator("lmsys/vicuna-13b-v1.3", load_in_8bit=True)

history = gen.multi_turn(
    texts,
    user_prefix="User: ",
    machine_prefix="Assistant: ",
    do_sample=True,
    top_p=0.9,
    temperature=0.3,
)

print(history)

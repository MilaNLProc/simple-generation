import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
from dataclasses import dataclass
from typing import Any, Dict
import click
from simple_generation import SimpleGenerator, DefaultGenerationConfig
import torch
import re
from langcodes.data_dicts import DEFAULT_SCRIPTS, LANGUAGE_ALPHA3
from langcodes import Language


@click.group()
def cli():
    pass


# def build_gui(type: str, generator, **generation_kwargs):
#     if type == "chat":
#         return ChatGUI(generator, generation_kwargs).build()
#     elif type == "translation":
#         return TranslationGUI(generator, generation_kwargs).build()
#     else:
#         raise ValueError(f"Unknown GUI type: {type}")


# @dataclass
# class ChatGUI:
#     generator: Any
#     generation_kwargs: Dict

#     def build(self):
#         def _chat(message, history):
#             messages = list()
#             for user_prompt, model_response in history:
#                 messages.append({"role": "user", "content": user_prompt})
#                 messages.append({"role": "assistant", "content": model_response})
#             messages.append({"role": "user", "content": message})

#             tokenized_chat = self.tokenizer.apply_chat_template(
#                 messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
#             ).to(self.device)

#             streamer = TextIteratorStreamer(
#                 self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
#             )
#             current_generation_args = self._prepare_generation_args(
#                 **self.generation_kwargs
#             )

#             gen_args = dict(
#                 inputs=tokenized_chat,
#                 streamer=streamer,
#                 **current_generation_args,
#             )

#             t = Thread(target=self.model.generate, kwargs=gen_args)
#             t.start()
#             partial_message = ""
#             for new_token in streamer:
#                 if new_token != "<":
#                     partial_message += new_token
#                     yield partial_message

#         interface = gr.ChatInterface(
#             _chat,
#             # chatbot=gr.Chatbot(height=300),
#             title=f"Chat with {self.model_name_or_path.split('/')[-1]}",
#             description="Generation arguments: " + str(self.generation_kwargs),
#             # fill_vertical_space=True, # this needs an upcoming gradio release
#         )
#         return interface


def split_sentences(text):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    return sentences


def get_opus_langs(model_name_or_path):
    langs = model_name_or_path.split("/")[-1].split("-")
    return langs[-2], langs[-1]


def prepare_generation(model_name_or_path, src_lang, tgt_lang, generator):
    additional_generation_kwargs = dict()

    # Model specific generation configurations
    if "nllb" in model_name_or_path:
        additional_generation_kwargs["forced_bos_token_id"] = (
            generator.tokenizer.lang_code_to_id[
                f"{LANGUAGE_ALPHA3[tgt_lang]}_{DEFAULT_SCRIPTS[tgt_lang]}"
            ]
        )

    return additional_generation_kwargs


def list_language_choices():
    return [(Language.get(l).display_name(), l) for l in DEFAULT_SCRIPTS.keys()]


@cli.command()
@click.option(
    "--model_name_or_path", "-m", type=str, default="facebook/nllb-200-distilled-600M"
)
def translation(model_name_or_path):

    generator = SimpleGenerator(model_name_or_path, torch_dtype=torch.bfloat16)

    is_opus = "opus-mt" in model_name_or_path
    if is_opus:
        opus_src_lang, opus_tgt_lang = get_opus_langs(model_name_or_path)
        print(f"OPUS model detected: {opus_src_lang} -> {opus_tgt_lang}")

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
                f"""
                ## Simple Generation Translation Interface
                        
                Model: **{model_name_or_path}**
            """
            )

        with gr.Accordion(label="Configuration", open=False):
            do_split_in_sentences = gr.Checkbox(value=True, label="Split in sentences.")
            do_sample = gr.Checkbox(value=False, label="do_sample")
            num_beams = gr.Number(value=1, label="num_beams", interactive=True)
            top_p = gr.Slider(
                value=0.9, label="top_p", minimum=0.1, maximum=1.0, interactive=True
            )
            top_k = gr.Number(value=50, label="top_k", interactive=True)
            temperature = gr.Slider(
                value=1.0,
                label="temperature",
                minimum=0.1,
                maximum=2.0,
                interactive=True,
            )

        def run_translation(
            src_lang,
            src_text,
            tgt_lang,
            do_sample,
            num_beams,
            top_p,
            top_k,
            temperature,
        ):

            texts = split_sentences(src_text) if do_split_in_sentences else src_text

            additional_generation_kwargs = prepare_generation(
                model_name_or_path, src_lang, tgt_lang, generator
            )
            generation_kwargs = {
                "do_sample": do_sample,
                "num_beams": num_beams,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
            }

            generation_kwargs.update(additional_generation_kwargs)

            print(generation_kwargs)

            outputs = generator(
                texts,
                skip_prompt=False,
                starting_batch_size=4,
                **additional_generation_kwargs,
            )

            return " ".join(outputs)

        with gr.Row():
            with gr.Column():
                src_lang = gr.Dropdown(
                    label="src_lang",
                    choices=list_language_choices(),
                    value=(opus_src_lang if is_opus else "en"),
                    show_label=False,
                    filterable=True,
                    interactive=False if is_opus else True,
                )
                src_text = gr.Textbox(
                    label="src_text",
                    placeholder="Enter text to translate",
                    lines=8,
                    interactive=True,
                    show_label=False,
                )
                btn = gr.Button("Translate", variant="primary")

            with gr.Column():
                tgt_lang = gr.Dropdown(
                    label="tgt_lang",
                    choices=list_language_choices(),
                    value=(opus_tgt_lang if is_opus else "it"),
                    filterable=True,
                    show_label=False,
                    interactive=False if is_opus else True,
                )
                tgt_text = gr.Textbox(
                    label="tgt_text",
                    lines=8,
                    interactive=False,
                    show_label=False,
                    show_copy_button=True,
                )

            btn.click(
                run_translation,
                inputs=[
                    src_lang,
                    src_text,
                    tgt_lang,
                    do_sample,
                    num_beams,
                    top_p,
                    top_k,
                    temperature,
                ],
                outputs=[tgt_text],
            )

    demo.launch()

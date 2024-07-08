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


@cli.command()
@click.option("--model_name_or_path", "-m", type=str, default="google/gemma-2-9b-it")
def chat(model_name_or_path):
    generator = SimpleGenerator(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    def _chat(
        message,
        history,
        do_sample,
        num_beams,
        top_p,
        top_k,
        temperature,
        max_new_tokens,
        add_generation_prompt,
    ):
        messages = list()
        for user_prompt, model_response in history:
            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": model_response})
        messages.append({"role": "user", "content": message})

        tokenized_chat = generator.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        ).to(generator.device)

        streamer = TextIteratorStreamer(
            generator.tokenizer,
            timeout=10.0,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        current_generation_args = generator._prepare_generation_args(
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # print("current_generation_args", current_generation_args)
        gen_args = dict(
            inputs=tokenized_chat,
            streamer=streamer,
            **current_generation_args,
        )

        t = Thread(target=generator.model.generate, kwargs=gen_args)
        t.start()
        partial_message = ""
        for new_token in streamer:
            if new_token != "<":
                partial_message += new_token
                yield partial_message

    with gr.Blocks(fill_height=True) as interface:
        gr.Markdown(
            f"""
                ## Simple Generation Chat Interface
                        
                Model: **{model_name_or_path}**
            """
        )

        with gr.Accordion(label="Configuration", open=False):
            do_sample = gr.Checkbox(value=True, label="do_sample", interactive=True)
            num_beams = gr.Number(value=1, label="num_beams", interactive=True)
            top_p = gr.Slider(
                value=0.9, label="top_p", minimum=0.1, maximum=1.0, interactive=True
            )
            top_k = gr.Number(value=100, label="top_k", interactive=True)
            temperature = gr.Slider(
                value=0.7,
                label="temperature",
                minimum=0.1,
                maximum=2.0,
                interactive=True,
            )
            max_new_tokens = gr.Number(
                value=128, label="max_new_tokens", interactive=True
            )

            add_generation_prompt = gr.Checkbox(
                value=True,
                label="add_generation_prompt",
                interactive=True,
            )

        gr.ChatInterface(
            _chat,
            # chatbot=gr.Chatbot(
            #     show_copy_button=True, render_markdown=True, bubble_full_width=False
            # ),
            additional_inputs=[
                do_sample,
                num_beams,
                top_p,
                top_k,
                temperature,
                max_new_tokens,
                add_generation_prompt,
            ],
        )
    interface.launch()


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


DEFAULT_PROMPT_TEMPLATE = """Translate the following text from {src_lang_name} into {tgt_lang_name}.
{src_lang_name}: {src_text}
{tgt_lang_name}:"""


@cli.command()
@click.option(
    "--model_name_or_path", "-m", type=str, default="facebook/nllb-200-distilled-600M"
)
def translation(model_name_or_path):

    generator = SimpleGenerator(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    is_opus = "opus-mt" in model_name_or_path
    if is_opus:
        opus_src_lang, opus_tgt_lang = get_opus_langs(model_name_or_path)
        print(f"OPUS model detected: {opus_src_lang} -> {opus_tgt_lang}")

    is_nllb = "nllb" in model_name_or_path

    is_neither_opus_nor_nllb = (not is_opus) and (not is_nllb)

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
                f"""
                ## Simple Generation Translation Interface
                        
                Model: **{model_name_or_path}**
            """
            )

        with gr.Accordion(label="Configuration", open=False):
            do_split_in_sentences = gr.Checkbox(
                value=True, label="Split in sentences.", interactive=True
            )
            do_sample = gr.Checkbox(value=False, label="do_sample", interactive=True)
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

            # Parameters for modern LMs
            propmt_template = gr.Textbox(
                label="Prompt template (specify {src_lang_name}, {tgt_lang_name}, {src_text})",
                value=DEFAULT_PROMPT_TEMPLATE,
                lines=3,
                interactive=is_neither_opus_nor_nllb,
                show_label=True,
            )
            apply_chat_template = gr.Checkbox(
                value=True,
                label="apply_chat_template",
                interactive=is_neither_opus_nor_nllb,
            )
            add_generation_prompt = gr.Checkbox(
                value=True,
                label="add_generation_prompt",
                interactive=is_neither_opus_nor_nllb,
            )

        def run_translation(
            src_lang,
            src_text,
            tgt_lang,
            do_split_in_sentences,
            do_sample,
            num_beams,
            top_p,
            top_k,
            temperature,
            prompt_template,
            apply_chat_template,
            add_generation_prompt,
        ):

            texts = split_sentences(src_text) if do_split_in_sentences else src_text
            texts = [t.strip() for t in texts if len(t)]

            if is_neither_opus_nor_nllb:
                src_lang_name = Language.get(src_lang).display_name()
                tgt_lang_name = Language.get(tgt_lang).display_name()

                texts = [
                    prompt_template.format(
                        **{
                            "src_lang_name": src_lang_name,
                            "tgt_lang_name": tgt_lang_name,
                            "src_text": t,
                        }
                    )
                    for t in texts
                ]

            additional_generation_kwargs = prepare_generation(
                model_name_or_path, src_lang, tgt_lang, generator
            )
            generation_kwargs = {
                "do_sample": do_sample,
                "num_beams": num_beams,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "apply_chat_template": apply_chat_template,
                "add_generation_prompt": add_generation_prompt,
            }

            generation_kwargs.update(additional_generation_kwargs)

            outputs = generator(
                texts,
                skip_prompt=True,
                max_new_tokens=256,
                batch_size="auto",
                starting_batch_size=2,
                **generation_kwargs,
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
                    do_split_in_sentences,
                    do_sample,
                    num_beams,
                    top_p,
                    top_k,
                    temperature,
                    propmt_template,
                    apply_chat_template,
                    add_generation_prompt,
                ],
                outputs=[tgt_text],
            )

    demo.launch()

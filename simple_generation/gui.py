import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
from dataclasses import dataclass
from typing import Any, Dict


def build_gui(type: str, generator, **generation_kwargs):
    if type == "chat":
        return ChatGUI(generator, generation_kwargs).build()
    elif type == "translation":
        return TranslationGUI(generator, generation_kwargs).build()
    else:
        raise ValueError(f"Unknown GUI type: {type}")


@dataclass
class ChatGUI:
    generator: Any
    generation_kwargs: Dict

    def build(self):
        def _chat(message, history):
            messages = list()
            for user_prompt, model_response in history:
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": model_response})
            messages.append({"role": "user", "content": message})

            tokenized_chat = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
            )
            current_generation_args = self._prepare_generation_args(
                **self.generation_kwargs
            )

            gen_args = dict(
                inputs=tokenized_chat,
                streamer=streamer,
                **current_generation_args,
            )

            t = Thread(target=self.model.generate, kwargs=gen_args)
            t.start()
            partial_message = ""
            for new_token in streamer:
                if new_token != "<":
                    partial_message += new_token
                    yield partial_message

        interface = gr.ChatInterface(
            _chat,
            # chatbot=gr.Chatbot(height=300),
            title=f"Chat with {self.model_name_or_path.split('/')[-1]}",
            description="Generation arguments: " + str(self.generation_kwargs),
            # fill_vertical_space=True, # this needs an upcoming gradio release
        )
        return interface


@dataclass
class TranslationGUI:
    generator: Any
    generation_kwargs: Dict

    def run_translation(self, src_lang, src_text, tgt_lang):
        texts = [s.strip() for s in src_text.split("\n")]

        return self.generator(
            texts,
            skip_prompt=False,
            **self.generation_kwargs,
        )

    def build(self):

        with gr.Blocks() as demo:

            with gr.Row():
                gr.Markdown("### Simple Generation Translation Interface")

            with gr.Row():
                with gr.Column():
                    src_lang = gr.Dropdown(
                        label="src_lang",
                        choices=[("English", "en"), ("Italian", "it")],
                        show_label=False,
                        filterable=True,
                    )
                    src_text = gr.Textbox(
                        label="src_text",
                        placeholder="Enter text to translate",
                        lines=8,
                        interactive=True,
                        show_label=False,
                    )
                    btn = gr.Button("Translate")

                with gr.Column():
                    tgt_lang = gr.Dropdown(
                        label="tgt_lang",
                        choices=[("English", "en"), ("Italian", "it")],
                        filterable=True,
                        show_label=False,
                    )
                    tgt_text = gr.Textbox(
                        label="tgt_text",
                        placeholder="Translation will appear",
                        lines=8,
                        interactive=False,
                        show_label=False,
                        show_copy_button=True,
                    )

            btn.click(
                self.run_translation,
                inputs=[src_lang, src_text, tgt_lang],
                outputs=tgt_text,
            )

        return demo

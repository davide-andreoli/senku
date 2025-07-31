import gradio as gr
import os
import re
from models.gpt import GPTModel
from tokenizer.tokenizer import Tokenizer


def see_available_checkpoints():
    filename_pattern = r"vocab(\d+).*emb(\d+).*ctx(\d+).*layers(\d+).*heads(\d+)"
    filename_regex = re.compile(filename_pattern)

    checkpoints = []
    for file in os.listdir("checkpoints"):
        if file.endswith(".pt"):
            match = filename_regex.search(file)
            if not match:
                continue
            vocab, emb, ctx, layers, heads = match.groups()
            checkpoints.append(
                f"Embedding size: {emb} - Context length: {ctx} - Layers: {layers} - Heads: {heads} - Vocabulary size: {vocab}"
            )
    return checkpoints


def reload_dropdown():
    return gr.update(choices=see_available_checkpoints())


def load_model(model_string: str):
    model_string_pattern = r"Embedding size: (\d+) - Context length: (\d+) - Layers: (\d+) - Heads: (\d+) - Vocabulary size: (\d+)"
    model_string_regex = re.compile(model_string_pattern)
    match = model_string_regex.search(model_string)
    if not match:
        return """Model not found!""", None, None
    emb, ctx, layers, heads, vocab = match.groups()
    filename = f"vocab{vocab}_emb{emb}_ctx{ctx}_layers{layers}_heads{heads}.pt"

    model_config = {
        "vocabulary_size": int(vocab),
        "embedding_dimension": int(emb),
        "context_length": int(ctx),
        "number_of_layers": int(layers),
        "number_of_attention_heads": int(heads),
        "dropout": 0.1,
        "bias": False,
    }
    model = GPTModel(**model_config)
    model.load_checkpoint(os.path.join("checkpoints", filename))
    tokenizer = Tokenizer()
    return (
        f"""Model loaded successfully!
    - Vocabulary size: {vocab}
    - Embedding size: {emb}
    - Context length: {ctx}
    - Layers: {layers}
    - Heads: {heads}\n
    This model has already been trained for {model.epochs} epochs.
""",
        model,
        tokenizer,
    )


def run_inference(
    model: GPTModel,
    tokenizer: Tokenizer,
    prompt: str,
    top_k: int,
    top_p: float,
    temperature: float,
    max_length: int,
    stop_at_eos: bool,
):
    haiku = model.generate_haiku(
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        stop_at_eos=stop_at_eos,
    )
    return haiku


with gr.Blocks() as inference:
    with gr.Row():
        gr.Markdown("# Inference")

    with gr.Row():
        gr.Markdown("## Model selection")

    select_model_dropdown = gr.Dropdown(
        choices=see_available_checkpoints(), label="Select model", interactive=True
    )
    refresh_model_dropdown = gr.Button("Refresh model list")

    model_state = gr.State()
    tokenizer_state = gr.State()

    with gr.Row():
        gr.Markdown("## Loaded model details")

    model_details = gr.Markdown(label="Model details")

    with gr.Row():
        gr.Markdown("## Inference settings")

    top_k = gr.Number(value=10, label="Top K")
    top_p = gr.Number(value=0.9, label="Top P")
    temperature = gr.Number(value=0.8, label="Temperature")
    max_length = gr.Number(value=100, label="Max length")
    stop_at_eos = gr.Checkbox(value=True, label="Stop at EOS")
    prompt = gr.Textbox(label="Prompt")
    generate_button = gr.Button("Generate")

    with gr.Row():
        gr.Markdown("## Inference output")

    inference_output = gr.Textbox(label="Inference output")

    refresh_model_dropdown.click(
        fn=reload_dropdown, inputs=[], outputs=[select_model_dropdown]
    )

    select_model_dropdown.change(
        fn=load_model,
        inputs=[select_model_dropdown],
        outputs=[model_details, model_state, tokenizer_state],
    )

    inference.load(
        fn=load_model,
        inputs=[select_model_dropdown],
        outputs=[model_details, model_state, tokenizer_state],
    )

    generate_button.click(
        fn=run_inference,
        inputs=[
            model_state,
            tokenizer_state,
            prompt,
            top_k,
            top_p,
            temperature,
            max_length,
            stop_at_eos,
        ],
        outputs=[inference_output],
    )


if __name__ == "__main__":
    inference.launch()

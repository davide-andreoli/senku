import gradio as gr
from core.inference import predict
from core.common import list_available_checkpoints


def gradio_list_available_checkpoints():
    checkpoints = list_available_checkpoints()
    checkpoint_dict = {str(checkpoint): checkpoint for checkpoint in checkpoints}
    return checkpoint_dict


def reload_dropdown():
    return gr.update(choices=list(gradio_list_available_checkpoints().keys()))


def load_model_details(checkpoint_key: str):
    checkpoint = gradio_list_available_checkpoints()[checkpoint_key]
    return checkpoint.checkpoint_details_string


def gradio_run_inference(
    checkpoint_key: str,
    prompt: str,
    top_k: int,
    top_p: float,
    temperature: float,
    max_length: int,
    stop_at_eos: bool,
):
    checkpoint = gradio_list_available_checkpoints()[checkpoint_key]
    return predict(
        checkpoint, prompt, top_k, top_p, temperature, max_length, stop_at_eos
    )


with gr.Blocks() as inference:
    with gr.Row():
        gr.Markdown("# Inference")

    with gr.Row():
        gr.Markdown("## Model selection")

    select_model_dropdown = gr.Dropdown(
        choices=list(gradio_list_available_checkpoints().keys()),
        label="Select model",
        interactive=True,
    )
    refresh_model_dropdown = gr.Button("Refresh model list")

    with gr.Row():
        gr.Markdown("## Loaded model details")

    model_details = gr.Markdown(label="Model details")

    with gr.Row():
        gr.Markdown("## Inference settings")

    top_k = gr.Number(
        value=10,
        label="Top K",
        info="Limits the number of possible next words the model considers. Lower values make the output more focused and deterministic.",
    )
    top_p = gr.Number(
        value=0.9,
        label="Top P",
        info="Controls how many possible next words are considered based on their combined probability. Lower values make the output more focused; higher values allow more creativity.",
    )
    temperature = gr.Number(
        value=0.8,
        label="Temperature",
        info="Controls the randomness of the output. Lower values make results more predictable; higher values add variety and creativity.",
    )
    max_length = gr.Number(
        value=100,
        label="Max length",
        info="The maximum number of words or tokens the model will generate in the response.",
    )
    stop_at_eos = gr.Checkbox(
        value=True,
        label="Stop at EOS",
        info="If checked, the model will stop generating text when it reaches an end-of-sentence token.",
    )
    prompt = gr.Textbox(
        label="Prompt", info="The input or question you want the AI to respond to."
    )
    generate_button = gr.Button("Generate")

    with gr.Row():
        gr.Markdown("## Inference output")

    inference_output = gr.Textbox(label="Inference output")

    refresh_model_dropdown.click(
        fn=reload_dropdown, inputs=[], outputs=[select_model_dropdown]
    )

    select_model_dropdown.change(
        fn=load_model_details,
        inputs=[select_model_dropdown],
        outputs=[model_details],
    )

    inference.load(
        fn=load_model_details,
        inputs=[select_model_dropdown],
        outputs=[model_details],
    )

    generate_button.click(
        fn=gradio_run_inference,
        inputs=[
            select_model_dropdown,
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

import gradio as gr
from core.inference import list_available_checkpoints, load_model, run_inference


def reload_dropdown():
    return gr.update(choices=list_available_checkpoints())


with gr.Blocks() as inference:
    with gr.Row():
        gr.Markdown("# Inference")

    with gr.Row():
        gr.Markdown("## Model selection")

    select_model_dropdown = gr.Dropdown(
        choices=list_available_checkpoints(), label="Select model", interactive=True
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

import gradio as gr
from core.train import validate_model, launch_training


def gradio_validate_model(
    embedding_dimension: int = 128,
    context_length: int = 64,
    num_layers: int = 8,
    num_heads: int = 8,
    dropout: float = 0.1,
    bias: bool = False,
):
    validation_output, validity = validate_model(
        embedding_dimension=embedding_dimension,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
    )
    return validation_output, gr.update(visible=validity)


with gr.Blocks() as train:
    with gr.Row():
        gr.Markdown("# Train")
    with gr.Column():
        gr.Markdown("## Model settings")
        # Model settings
        embedding_dimension = gr.Number(value=128, label="Embedding dimension")
        context_length = gr.Number(value=128, label="Context length")
        num_layers = gr.Number(value=8, label="Number of layers")
        num_heads = gr.Number(value=8, label="Number of attention heads")
        dropout = gr.Number(value=0.1, label="Dropout")
        bias = gr.Checkbox(value=False, label="Use bias")
        validation_output = gr.Markdown(label="Model validation")

    with gr.Column():
        gr.Markdown("## Training settings")
        num_epochs = gr.Number(value=50, label="Number of epochs")
        batch_size = gr.Number(value=32, label="Batch size")
        reset = gr.Checkbox(value=False, label="Reset checkpoint")
        start_training_button = gr.Button("Start training")

    with gr.Row():
        gr.Markdown("## Training progress")

    training_output = gr.Markdown(
        "Training outcome will be shown here", label="Model validation"
    )

    model_settings = [
        embedding_dimension,
        context_length,
        num_layers,
        num_heads,
        dropout,
        bias,
    ]

    for component in model_settings:
        component.change(
            fn=gradio_validate_model,
            inputs=model_settings,
            outputs=[validation_output, start_training_button],
        )

    start_training_button.click(
        fn=launch_training,
        inputs=[
            embedding_dimension,
            context_length,
            num_layers,
            num_heads,
            dropout,
            bias,
            num_epochs,
            batch_size,
            reset,
        ],
        outputs=[training_output],
    )

if __name__ == "__main__":
    train.launch()

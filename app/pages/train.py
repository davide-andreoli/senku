import gradio as gr
from core.train import validate_model, launch_training, resume_training
from core.common import list_available_checkpoints


def gradio_list_available_checkpoints():
    checkpoints = list_available_checkpoints()
    checkpoint_dict = {str(checkpoint): checkpoint for checkpoint in checkpoints}
    return checkpoint_dict


def gradio_resume_training(checkpoint_key, res_num_epochs, res_batch_size):
    checkpoint = gradio_list_available_checkpoints()[checkpoint_key]
    result = resume_training(checkpoint, res_num_epochs, res_batch_size)
    return result


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


def load_model_details(checkpoint_key):
    checkpoint = gradio_list_available_checkpoints()[checkpoint_key]
    return checkpoint.checkpoint_details_string


with gr.Blocks() as train:
    with gr.Row():
        gr.Markdown("# Train")
    with gr.Tab("Existing model"):
        gr.Markdown("## Train an existing model")

        select_model_dropdown = gr.Dropdown(
            choices=list(gradio_list_available_checkpoints().keys()),
            label="Select model",
            interactive=True,
        )

        with gr.Column():
            gr.Markdown("## Selected model details")
            selected_model_output = gr.Markdown(label="Model details")

        with gr.Column():
            gr.Markdown("## Training settings")
            res_num_epochs = gr.Number(value=50, label="Number of epochs")
            res_batch_size = gr.Number(value=32, label="Batch size")
            resume_training_button = gr.Button("Resume training")

    with gr.Tab("New model"):
        gr.Markdown("## Train a new model")
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

    select_model_dropdown.change(
        fn=load_model_details,
        inputs=[select_model_dropdown],
        outputs=[selected_model_output],
    )
    train.load(
        fn=load_model_details,
        inputs=[select_model_dropdown],
        outputs=[selected_model_output],
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
        ],
        outputs=[training_output],
    )

    resume_training_button.click(
        fn=gradio_resume_training,
        inputs=[select_model_dropdown, res_num_epochs, res_batch_size],
        outputs=[training_output],
    )

if __name__ == "__main__":
    train.launch()

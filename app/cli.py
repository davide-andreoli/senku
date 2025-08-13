import click
from app.core.dataset import load_default_dataset
from app.core.train import validate_model, launch_training, resume_training
from app.core.inference import predict
from app.core.common import list_available_checkpoints, get_checkpoint
from app.core.rich import display_training_progress, display_table_sample
from rich.console import Console


console = Console()


@click.group()
def cli():
    pass


@cli.command()
def tui():
    """Launch the Text User Interface"""
    from app.tui_app import run_app

    run_app()


@cli.command()
def gui():
    """Launch the Graphical User Interface"""
    from app.main import senku_app

    senku_app.launch()


@cli.command(name="load-default-dataset")
def cli_load_default_dataset():
    """Load the default dataset into the dataset/valid-haikus.csv file. This file will be used for training."""
    stats, sample, _ = load_default_dataset()
    console.print(stats)
    display_table_sample(sample)


@cli.command()
@click.option("--checkpoint", prompt="Checkpoint", help="Use output from list-models.")
@click.option("--epochs", default=50, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
def train_existing_model(
    checkpoint: str,
    epochs: int,
    batch_size: int,
):
    """Resume the training of an existing model."""
    senku_checkpoint = get_checkpoint(checkpoint)
    trainig_output = resume_training(
        checkpoint=senku_checkpoint,
        num_epochs=epochs,
        batch_size=batch_size,
    )
    display_training_progress(trainig_output)


@cli.command()
@click.option(
    "--tokenizer-strategy",
    type=click.Choice(["character", "syllable", "word"], case_sensitive=False),
    default="character",
    show_default=True,
)
@click.option("--embedding-dimension", default=128, show_default=True)
@click.option("--context-length", default=128, show_default=True)
@click.option("--num-layers", default=8, show_default=True)
@click.option("--num-heads", default=8, show_default=True)
@click.option("--dropout", default=0.1, show_default=True)
@click.option("--bias", is_flag=True, show_default=True)
@click.option("--epochs", default=50, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
@click.option("--checkpoint-name", default="")
def train_new_model(
    tokenizer_strategy: str,
    embedding_dimension: int,
    context_length: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    bias: bool,
    epochs: int,
    batch_size: int,
    checkpoint_name: str,
):
    """Train a new model from scratch."""
    validation_output, validity = validate_model(
        embedding_dimension=embedding_dimension,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
        tokenizer_strategy=tokenizer_strategy,
    )
    if not validity:
        console.print(validation_output)
        return
    training_output = launch_training(
        embedding_dimension=embedding_dimension,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
        num_epochs=epochs,
        batch_size=batch_size,
        tokenizer_strategy=tokenizer_strategy,
        checkpoint_name=checkpoint_name,
    )
    display_training_progress(training_output)


@cli.command()
def list_models():
    """List all models that have been trained."""
    for model in list_available_checkpoints():
        console.print(model)


@cli.command()
@click.option("--checkpoint", prompt="Checkpoint", help="Use output from list-models.")
@click.option("--prompt", default="", help="Haiku prompt")
@click.option("--top-k", default=10, show_default=True)
@click.option("--top-p", default=0.9, show_default=True)
@click.option("--temperature", default=0.8, show_default=True)
@click.option("--max-length", default=100, show_default=True)
@click.option("--stop-at-eos", is_flag=True, default=True, show_default=True)
def generate(
    checkpoint: str,
    prompt: str,
    top_k: int,
    top_p: float,
    temperature: float,
    max_length: int,
    stop_at_eos: bool,
):
    """Generate text using a trained model."""
    senku_checkpoint = get_checkpoint(checkpoint)
    haiku = predict(
        checkpoint=senku_checkpoint,
        prompt=prompt,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        stop_at_eos=stop_at_eos,
    )

    console.print("\nGenerated Haiku:\n")
    console.print(haiku)


if __name__ == "__main__":
    cli()

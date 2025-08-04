import click
from core.dataset import load_default_dataset
from core.train import validate_model, launch_training, resume_training
from core.inference import predict
from core.common import list_available_checkpoints, get_checkpoint
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli():
    pass


@cli.command(name="load-default-dataset")
def cli_load_default_dataset():
    stats, sample, _ = load_default_dataset()
    console.print(stats)
    table = Table(title="Sample")
    table.add_column("First line")
    table.add_column("Second line")
    table.add_column("Third line")

    for _, row in sample.iterrows():
        table.add_row(row["first_line"], row["second_line"], row["third_line"])
    console.print(table)


@cli.command()
@click.option("--checkpoint", prompt="Checkpoint", help="Use output from list-models.")
@click.option("--epochs", default=50)
@click.option("--batch-size", default=32)
def train_existing_model(
    checkpoint: str,
    epochs: int,
    batch_size: int,
):
    senku_checkpoint = get_checkpoint(checkpoint)
    trainig_output = resume_training(
        checkpoint=senku_checkpoint,
        num_epochs=epochs,
        batch_size=batch_size,
    )
    console.print(trainig_output)


@cli.command()
@click.option("--embedding-dimension", default=128)
@click.option("--context-length", default=128)
@click.option("--num-layers", default=8)
@click.option("--num-heads", default=8)
@click.option("--dropout", default=0.1)
@click.option("--bias", is_flag=True)
@click.option("--epochs", default=50)
@click.option("--batch-size", default=32)
def train_new_model(
    embedding_dimension: int,
    context_length: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    bias: bool,
    epochs: int,
    batch_size: int,
):
    validation_output, validity = validate_model(
        embedding_dimension=embedding_dimension,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
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
    )
    console.print(training_output)


@cli.command()
def list_models():
    for model in list_available_checkpoints():
        console.print(model)


@cli.command()
@click.option("--checkpoint", prompt="Checkpoint", help="Use output from list-models.")
@click.option("--prompt", default="", help="Haiku prompt")
@click.option("--top-k", default=10)
@click.option("--top-p", default=0.9)
@click.option("--temperature", default=0.8)
@click.option("--max-length", default=100)
@click.option("--stop-at-eos", is_flag=True, default=True)
def generate(
    checkpoint: str,
    prompt: str,
    top_k: int,
    top_p: float,
    temperature: float,
    max_length: int,
    stop_at_eos: bool,
):
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

import click
from core.dataset import load_default_dataset
from core.train import validate_model, launch_training
from core.inference import list_available_checkpoints, load_model, run_inference


@click.group()
def cli():
    pass


@cli.command(name="load-default-dataset")
def cli_load_default_dataset():
    stats, sample, _ = load_default_dataset()
    click.echo(stats)
    click.echo(sample.to_string(index=False))


@cli.command()
@click.option("--embedding-dimension", default=128)
@click.option("--context-length", default=128)
@click.option("--num-layers", default=8)
@click.option("--num-heads", default=8)
@click.option("--dropout", default=0.1)
@click.option("--bias", is_flag=True)
@click.option("--epochs", default=50)
@click.option("--batch-size", default=32)
@click.option("--reset", is_flag=True)
def train_model(
    embedding_dimension,
    context_length,
    num_layers,
    num_heads,
    dropout,
    bias,
    epochs,
    batch_size,
    reset,
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
        click.echo(validation_output)
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
        reset=reset,
    )
    click.echo(training_output)


@cli.command()
def list_models():
    for model in list_available_checkpoints():
        click.echo(model)


@cli.command()
@click.option("--model", prompt="Model string", help="Use output from list-models.")
@click.option("--prompt", default="", help="Haiku prompt")
@click.option("--top-k", default=10)
@click.option("--top-p", default=0.9)
@click.option("--temperature", default=0.8)
@click.option("--max-length", default=100)
@click.option("--stop-at-eos", is_flag=True, default=True)
def generate(model, prompt, top_k, top_p, temperature, max_length, stop_at_eos):
    msg, model_obj, tokenizer = load_model(model)
    if not model_obj:
        click.echo(msg)
        return

    haiku = run_inference(
        model=model_obj,
        tokenizer=tokenizer,
        prompt=prompt,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        stop_at_eos=stop_at_eos,
    )

    click.echo("\nGenerated Haiku:\n")
    click.echo(haiku)


if __name__ == "__main__":
    cli()

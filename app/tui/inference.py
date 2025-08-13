import questionary
from app.core.common import list_available_checkpoints
from app.core.inference import predict
from rich import print


def inference_flow():
    checkpoints = list_available_checkpoints()
    if not checkpoints:
        print("No checkpoints available. Train a model first.")
        return

    choices = [
        questionary.Choice(title=str(checkpoint), value=checkpoint)
        for checkpoint in checkpoints
    ]

    chosen_checkpoint = questionary.select(
        "Select a model checkpoint:", choices=choices
    ).ask()

    prompt = questionary.text("Prompt").ask()
    # top_k = int(questionary.text("Top K", default="10").ask())
    top_p = float(questionary.text("Top P", default="0.9").ask())
    temp = float(questionary.text("Temperature", default="0.8").ask())
    max_len = int(questionary.text("Max length", default="100").ask())
    stop = questionary.confirm("Stop at EOS?", default=True).ask()

    print("\nGenerating haiku...\n")
    haiku = predict(chosen_checkpoint, prompt, 10, top_p, temp, max_len, stop)
    print("\nGenerated Haiku:\n")
    print(haiku)

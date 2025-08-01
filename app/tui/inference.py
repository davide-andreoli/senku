import questionary
from core.inference import list_available_checkpoints, load_model
from rich import print


def inference_flow():
    checkpoints = list_available_checkpoints()
    if not checkpoints:
        print("No checkpoints available. Train a model first.")
        return

    model_str = questionary.select(
        "Select a model checkpoint:", choices=checkpoints
    ).ask()
    msg, model, tokenizer = load_model(model_str)
    if model is None:
        print(msg)
        return

    print(msg)
    prompt = questionary.text("Prompt").ask()
    # top_k = int(questionary.text("Top K", default="10").ask())
    top_p = float(questionary.text("Top P", default="0.9").ask())
    temp = float(questionary.text("Temperature", default="0.8").ask())
    max_len = int(questionary.text("Max length", default="100").ask())
    stop = questionary.confirm("Stop at EOS?", default=True).ask()

    print("\nGenerating haiku...\n")
    haiku = model.generate_haiku(
        tokenizer=tokenizer,
        prompt=prompt,
        top_p=top_p,
        temperature=temp,
        max_length=max_len,
        stop_at_eos=stop,
    )
    print("\nGenerated Haiku:\n")
    print(haiku)

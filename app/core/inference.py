import os
import re
from models.gpt import GPTModel
from tokenizer.tokenizer import CharacterTokenizer
from helpers.checkpoint import SenkuCheckpoint


def list_available_checkpoints():
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
    tokenizer = CharacterTokenizer()
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


def predict(
    checkpoint: SenkuCheckpoint,
    prompt: str,
    top_k: int,
    top_p: float,
    temperature: float,
    max_length: int,
    stop_at_eos: bool,
):
    model = checkpoint.instantiate_model()
    tokenizer = checkpoint.instantiate_tokenizer()
    return run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        stop_at_eos=stop_at_eos,
    )


def run_inference(
    model: GPTModel,
    tokenizer: CharacterTokenizer,
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

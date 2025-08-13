# senku: A Tiny Transformer for Generating Haikus

> In the quiet code,  
> ideas bloom in three short lines.  
> Haikus from machines.

## Project Overview

Senku is a lightweight transformer-based language model trained to generate haikus, short structured poems in 5-7-5 syllable format. It is designed to be trained and run on a CPU, so the performances will be mediocre, but the aim is to explores minimal language model design, tokenization, training loops, and inference with a focus on tiny-scale LLM experimentation.

### Goals

- Understand transformer internals from scratch
- Implement a simple tokenizer and attention mechanism
- Keep the model small enough to train on a laptop
- Train on a dataset of haikus (plain text or structured CSV)
- Generate haikus from prompts using sampling strategies

### Project Structure

## Details

### Model Architecture

- Character-level tokenizer (with <PAD>, <EOS>, <UNK>)
- GPT-style decoder-only transformer
- Configurable parameters:
- Embedding size
- Number of layers
- Attention heads
- Context length

### Dataset Format

Haikus are stored in CSV format as three separate lines per row:

```csv
line1,line2,line3
An old silent pond,A frog jumps into the pond,Splash! Silence again.
```

Each haiku is tokenized and truncated to the context_length if necessary.

## Roadmap

- Add the possibility to experiment with different model architectures
- Add instruction fine tuning
- Add an image generation part

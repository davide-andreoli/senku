import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, context_length: int, dropout: float, number_of_heads: int, bias: bool = False):
        super().__init__()
        if output_dimension % number_of_heads != 0:
            raise ValueError("The output dimension must be divisible by the number of heads.")
        
        self.output_dimension = output_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = output_dimension // number_of_heads

        # Initialize the linear layers for queries, keys, and values
        self.weights_query = nn.Linear(input_dimension, output_dimension, bias=bias)
        self.weights_key = nn.Linear(input_dimension, output_dimension, bias=bias)
        self.weights_value = nn.Linear(input_dimension, output_dimension, bias=bias)
        self.output_projection = nn.Linear(output_dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_dimension, number_of_tokens, input_dimension = x.shape
        keys = self.weights_key(x)
        queries = self.weights_query(x)
        values = self.weights_value(x)

        keys = keys.view(batch_dimension, number_of_tokens, self.number_of_heads, self.head_dimension)
        values = values.view(batch_dimension, number_of_tokens, self.number_of_heads, self.head_dimension)
        queries = queries.view(batch_dimension, number_of_tokens, self.number_of_heads, self.head_dimension)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:number_of_tokens, :number_of_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)

        context_vector = context_vector.contiguous().view(batch_dimension, number_of_tokens, self.output_dimension)
        context_vector = self.output_projection(context_vector)
        return context_vector
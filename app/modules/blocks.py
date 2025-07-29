import torch
from torch import nn
from modules.attention import MultiHeadAttention
from modules.normalization import LayerNorm
from modules.activation import GELU

class FeedForward(nn.Module):
    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dimension, 4 * embedding_dimension),
            GELU(),
            nn.Linear(4 * embedding_dimension, embedding_dimension),
        )
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension: int, number_of_heads: int, context_length: int, dropout: float, bias: bool):
        super().__init__()
        self.att = MultiHeadAttention(input_dimension=embedding_dimension, output_dimension=embedding_dimension,context_length=context_length,number_of_heads=number_of_heads,dropout=dropout,bias=bias)
        self.ff = FeedForward(embedding_dimension=embedding_dimension)
        self.norm1 = LayerNorm(embedding_dimension)
        self.norm2 = LayerNorm(embedding_dimension)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
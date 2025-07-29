import torch
from torch import nn
from modules.blocks import TransformerBlock
from modules.normalization import LayerNorm

class GPTModel(nn.Module):

    def __init__(self, vocabulary_size: int, embedding_dimension: int, context_length: int, number_of_layers: int, dropout: float, bias: bool, number_of_attention_heads: int):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.position_embedding = nn.Embedding(context_length, embedding_dimension)
        self.dropout_embedding = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(
        *[TransformerBlock(embedding_dimension=embedding_dimension, number_of_heads = number_of_attention_heads, context_length = context_length, dropout = dropout, bias = bias) for _ in range(number_of_layers)])
        self.final_normalization = LayerNorm(embedding_dimension)
        self.output_head = nn.Linear(
        embedding_dimension, vocabulary_size, bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        position_embeddings = self.position_embedding(torch.arange(seq_len, device=in_idx.device))
        x = token_embeddings + position_embeddings
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_normalization(x)
        logits = self.output_head(x)
        return logits
    
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def total_size(self, unit: str = 'MB') -> float:
        total_parameters = self.total_parameters()
        size_bytes = total_parameters * 4
        if unit == 'MB':
            size = size_bytes / (1024 ** 2)
        elif unit == 'GB':
            size = size_bytes / (1024 ** 3)
        else:
            raise ValueError("Unit must be either 'MB' or 'GB'.")
        return size
    
    def generate_tokens(self, indexes: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, top_p: float = None, do_sample: bool = True) -> torch.Tensor:
        """
        Generate tokens with various sampling strategies
        
        Args:
            indexes: Initial token sequence [batch_size, seq_len]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only consider top k tokens for sampling
            top_p: Nucleus sampling - consider tokens with cumulative probability up to p
            do_sample: If False, use greedy decoding
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                indexes_to_consider = indexes[:, -self.context_length:]
                
                logits = self(indexes_to_consider)
                logits = logits[:, -1, :] 
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                if do_sample:
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_index = torch.multinomial(probs, num_samples=1)
                else:
                    next_index = torch.argmax(logits, dim=-1, keepdim=True)
                
                indexes = torch.cat((indexes, next_index), dim=1)
        
        return indexes
    
    def generate_haiku(self, tokenizer, prompt: str = "", max_length: int = 100, temperature: float = 0.8, top_p: float = 0.9, stop_at_eos: bool = True):
        """
        Generate a haiku with proper formatting
        
        Args:
            tokenizer: Tokenizer instance
            prompt: Optional prompt to start generation
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop_at_eos: Whether to stop at EOS token
        """
        self.eval()
        
        if prompt:
            input_ids = tokenizer.encode_to_tensor(prompt)
        else:
            input_ids = torch.tensor([[]], dtype=torch.long)
            if input_ids.size(1) == 0:
                input_ids = torch.tensor([[tokenizer.encode_dict.get(' ', 0)]], dtype=torch.long)
        
        generated = self.generate_tokens(
            input_ids, 
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        
        generated_text = tokenizer.decode_from_tensor(generated)
        
        if stop_at_eos and "<EOS>" in generated_text:
            generated_text = generated_text.split("<EOS>")[0]
        
        if prompt:
            generated_text = generated_text[len(prompt):]
        
        return generated_text.strip()
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])


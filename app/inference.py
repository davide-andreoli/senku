import torch
import argparse
import os
from models.gpt import GPTModel
from tokenizer.tokenizer import Tokenizer
import json
import re

class HaikuGenerator:
    def __init__(self, model_path: str, model_config: dict = None):
        """
        Initialize haiku generator
        
        Args:
            model_path: Path to model checkpoint
            model_config: Model configuration dict
        """
        self.tokenizer = Tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default model config for small laptop-friendly model
        if model_config is None:
            model_config = {
                'vocabulary_size': self.tokenizer.vocabulary_size,
                'embedding_dimension': 128,
                'context_length': 64,
                'number_of_layers': 6,
                'number_of_attention_heads': 8,
                'dropout': 0.1,
                'bias': False
            }
        
        self.model = GPTModel(**model_config)
        
        if os.path.exists(model_path):
            self.model.load_checkpoint(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model checkpoint {model_path} not found. Using untrained model.")
            
        self.model.to(self.device)
        self.model.eval()
    
    def generate_haiku(self, prompt: str = "", num_samples: int = 1, temperature: float = 0.8, top_p: float = 0.9, max_length: int = 80):
        """Generate haiku(s) with optional prompt"""
        haikus = []
        
        for _ in range(num_samples):
            haiku = self.model.generate_haiku(
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            haikus.append(self._format_haiku(haiku))
        
        return haikus if num_samples > 1 else haikus[0]
    
    def _format_haiku(self, text: str) -> str:
        """Clean and format generated haiku"""
        text = text.replace("<UNK>", "").replace("<PAD>", "").replace("<EOS>", "")
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        lines = text.strip().split('\n')
        if len(lines) == 1:
            words = text.split()
            if len(words) > 6:
                third = len(words) // 3
                lines = [
                    ' '.join(words[:third]),
                    ' '.join(words[third:2*third]),
                    ' '.join(words[2*third:])
                ]
        
        while len(lines) < 3:
            lines.append("")
        lines = lines[:3]
        
        return '\n'.join(line.strip() for line in lines if line.strip())
    
    def evaluate_diversity(self, num_samples: int = 10, prompt: str = ""):
        """Evaluate diversity of generated haikus"""
        haikus = [self.generate_haiku(prompt=prompt) for _ in range(num_samples)]
        
        unique_haikus = len(set(haikus))
        diversity_score = unique_haikus / num_samples
        
        return {
            'diversity_score': diversity_score,
            'unique_haikus': unique_haikus,
            'total_samples': num_samples,
            'samples': haikus
        }
    
    def interactive_mode(self):
        """Interactive haiku generation"""
        print("Haiku Gen - Interactive Mode")
        print("Type 'quit' to exit, 'settings' to adjust parameters")
        print("=" * 50)
        
        temperature = 0.8
        top_p = 0.9
        num_samples = 1
        
        while True:
            try:
                user_input = input("\nEnter a prompt (or press Enter for random): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'settings':
                    print(f"Current settings:")
                    print(f"  Temperature: {temperature}")
                    print(f"  Top-p: {top_p}")
                    print(f"  Number of samples: {num_samples}")
                    
                    try:
                        new_temp = input(f"New temperature ({temperature}): ").strip()
                        if new_temp:
                            temperature = float(new_temp)
                        
                        new_top_p = input(f"New top-p ({top_p}): ").strip()
                        if new_top_p:
                            top_p = float(new_top_p)
                        
                        new_samples = input(f"Number of samples ({num_samples}): ").strip()
                        if new_samples:
                            num_samples = int(new_samples)
                    except ValueError:
                        print("Invalid input, keeping current settings")
                    continue
                
                print("\nGenerated Haiku(s):")
                print("-" * 30)
                
                haikus = self.generate_haiku(
                    prompt=user_input,
                    num_samples=num_samples,
                    temperature=temperature,
                    top_p=top_p
                )
                
                if isinstance(haikus, list):
                    for i, haiku in enumerate(haikus, 1):
                        print(f"\n{i}:")
                        print(haiku)
                else:
                    print(haikus)
                
                print("-" * 30)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate haikus with trained model')
    parser.add_argument('--model_path', default='checkpoints/checkpoint.pt', help='Path to model checkpoint')
    parser.add_argument('--prompt', default='', help='Prompt for haiku generation')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of haikus to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling threshold')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model diversity')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--config', help='Path to model config JSON file')
    
    args = parser.parse_args()
    
    model_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            model_config = json.load(f)
    
    generator = HaikuGenerator(args.model_path, model_config)
    
    if args.interactive:
        generator.interactive_mode()
    elif args.evaluate:
        print("Evaluating model diversity...")
        results = generator.evaluate_diversity(num_samples=20, prompt=args.prompt)
        print(f"Diversity Score: {results['diversity_score']:.3f}")
        print(f"Unique Haikus: {results['unique_haikus']}/{results['total_samples']}")
        print("\nSample Haikus:")
        for i, haiku in enumerate(results['samples'][:5], 1):
            print(f"\n{i}:")
            print(haiku)
    else:
        haikus = generator.generate_haiku(
            prompt=args.prompt,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print("🌸 Generated Haiku(s):")
        print("=" * 30)
        
        if isinstance(haikus, list):
            for i, haiku in enumerate(haikus, 1):
                print(f"\n{i}:")
                print(haiku)
        else:
            print(haikus)
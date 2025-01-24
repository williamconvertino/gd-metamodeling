from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config.get_name()

    def forward(self, x):
        raise NotImplementedError

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_num_params_formatted(self):
        num_params = self.get_num_params()
        return f"{num_params / 1e6:.2f}M" if num_params > 1e6 else num_params

    def resize_vocabulary(self, d_vocab_new):
        self.config.d_vocab = d_vocab_new
        if hasattr(self, 'wte'):
            self.wte = nn.Embedding(d_vocab_new, self.config.d_embed)

    def generate(self, x, max_new_tokens=100, eos_token_id=None, top_k=10, temperature=1.0):
        
        input_size = x.size(1)

        for _ in range(max_new_tokens):
        
            logits, _ = self(x)
            logits = logits[:, -1, :]
            
            # Select next token based on top-k and temperature
            x_new = torch.multinomial(torch.softmax(logits / temperature, dim=-1), top_k)
            
            x = torch.cat((x, x_new), dim=1)
        
            if eos_token_id is not None and x_new.item() == eos_token_id:
                break

        x = x[:, input_size:]
        
        return x

    def beam_search(self, x, max_new_tokens=100, num_beams=3, eos_token_id=None, ngram_removal=3):
        
        input_size = x.size(1)

        beams = [{'x': x, 'score': 0, 'eos': False}]  # Initial beam
        
        for _ in range(max_new_tokens):
            
            new_sequences = []
            
            for beam in beams:
            
                # If EOS is already encountered, propagate the beam without changes
                if beam['eos']:
                    new_sequences.append(beam)
                    continue
                
                # Generate beam candidates
                logits, _ = self(beam['x'])
                topk = torch.topk(logits[:, -1, :], num_beams, dim=-1)
                
                for i in range(num_beams):
                    x = beam['x']
                    score = beam['score']
                    eos = beam['eos']
                    
                    next_idx = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
                    next_score = topk.values[0, i].item()
                    
                    if next_idx == eos_token_id:
                        eos = True
                    else:
                        x = torch.cat((x, next_idx), dim=1)
                        score += next_score
                    
                    new_sequences.append({
                        'x': x,
                        'score': score,
                        'eos': eos
                    })
                
            def has_repeated_ngram(sequence, n):
                ngrams = set()
                for i in range(len(sequence) - n + 1):
                    ngram = tuple(sequence[i:i + n].tolist())
                    if ngram in ngrams:
                        return True
                    ngrams.add(ngram)
                return False
            
            # Remove n-grams
            if ngram_removal is not None:
                ngram_removed_sequences = [seq for seq in new_sequences if not has_repeated_ngram(seq['x'][0], ngram_removal)]
                if len(ngram_removed_sequences) > 0:
                    new_sequences = ngram_removed_sequences
            
            # Select beam based on normalized score
            new_sequences.sort(key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1), reverse=True)
            beams = new_sequences[:num_beams]
            
            # Break early if all beams have encountered EOS
            if all(beam['eos'] for beam in beams):
                break
        
        most_probable_sequence = max(beams, key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1))
        
        x = most_probable_sequence['x']

        return x[:, input_size:]
    
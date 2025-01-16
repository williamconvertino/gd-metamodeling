from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class BaseConfig:

    # Model Name
    model_name: str

    # Basic Model Parameters
    d_vocab: int = 50258 # Default value for GPT-2 tokenizer
    d_seq: int = 256
    d_embed: int = 256
    n_head: int = 8
    n_layer: int = 1

    # Attention
    attn_fn: str = "softmax"

    # Additional Mechanisms
    use_ff: bool = True

    # Regularization and Normalization
    dropout: float = 0.1
    
    def get_name(self):
        return f"{self.model_name}_{self.d_seq}C_{self.d_embed}E_{self.n_head}H_{self.n_layer}L"
    
    def __post_init__(self):
        assert self.attn_fn in ["softmax", "linear", "rbf"], f"Invalid attention function ({self.attn_fn}), must be one of ['softmax', 'linear', 'rbf']"

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

    def generate(self, x, max_new_tokens=100, eos_token=None, return_inputs=False):
        
        input_size = x.size(1)

        for _ in range(max_new_tokens):
        
            logits, _ = self(x)
            x_new = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            x = torch.cat((x, x_new), dim=1)
        
            if eos_token is not None and x_new.item() == eos_token:
                break

        if not return_inputs:
            x = x[:, input_size:]
        
        return x

    def beam_search(self, x, max_new_tokens=100, num_beams=3, eos_token=None, return_inputs=False):
        
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
                    
                    if next_idx == eos_token:
                        eos = True
                    else:
                        x = torch.cat((x, next_idx), dim=1)
                        score += next_score
                    
                    new_sequences.append({
                        'x': x,
                        'score': score,
                        'eos': eos
                    })
                
            # Select beam based on normalized score
            new_sequences.sort(key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1), reverse=True)
            beams = new_sequences[:num_beams]
            
            # Break early if all beams have encountered EOS
            if all(beam['eos'] for beam in beams):
                break
        
        most_probable_sequence = max(beams, key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1))
        
        x = most_probable_sequence['x']

        if not return_inputs:
            x = x[:, input_size:]
        
        return x
    
    
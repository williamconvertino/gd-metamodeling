import setup_paths
import torch
from script_util import get_model_from_args, load_checkpoint, get_flags_from_args, get_models_from_name
from src.datasets import get_dataloaders, get_tokenizer
from src.evaluation import evaluate_models

if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    models = get_models_from_name()
    
    for model in models:
        checkpoint = load_checkpoint(model)
        assert checkpoint is not None, f"No checkpoint found for model [{model.config.get_name()}]"
        model.load_state_dict(checkpoint['model_state_dicts'][checkpoint['epoch']])
    
        print(f'Loaded model [{model.config.get_name()}] with [{checkpoint["epoch"]}] epochs')
    
    flags = get_flags_from_args()
    
    use_beam = 'use_beam' in flags
    
    tokenizer = get_tokenizer()
    
    print('Evaluating models using beam search' if use_beam else 'Evaluating models using top-k search')
    
    dataloaders = get_dataloaders(d_seq=model.config.d_seq, tokenizer=tokenizer, batch_size=64)
    evaluate_models(models, tokenizer, dataloaders, use_beam=use_beam)
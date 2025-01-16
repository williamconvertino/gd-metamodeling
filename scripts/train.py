import setup_paths
import torch
from script_util import get_model_from_args, load_checkpoint, get_flags_from_args
from src.training import train_model
from src.datasets import get_dataloaders, get_tokenizer
from src.models import GDConfig
from src.util.cache import setup_cache

if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    model = get_model_from_args()
    checkpoint = load_checkpoint(model)
    
    flags = get_flags_from_args()
    if 'cache' in flags:
        print('Setting up cache')
        setup_cache()
    
    tokenizer = get_tokenizer()
    model.resize_vocabulary(len(tokenizer))
    
    dataloaders = get_dataloaders(d_seq=model.config.d_seq, tokenizer=tokenizer, batch_size=64)
    train_model(model, tokenizer, dataloaders, checkpoint=checkpoint)
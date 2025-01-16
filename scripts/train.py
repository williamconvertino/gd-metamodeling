import setup_paths
import torch
from script_util import get_model_from_args, load_checkpoint
from src.training import train_model
from src.datasets import get_dataloaders, get_tokenizer
from src.models import GDConfig

if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    model = get_model_from_args()
    checkpoint = load_checkpoint(model)
    
    tokenizer = get_tokenizer()
    model.resize_vocabulary(len(tokenizer))
    
    dataloaders = get_dataloaders(tokenizer, d_seq=model.config.d_seq)
    train_model(model, tokenizer, dataloaders, checkpoint=checkpoint)
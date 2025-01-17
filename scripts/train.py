import setup_paths
import torch
import os
from script_util import get_model_from_args, load_checkpoint, get_flags_from_args
from src.training import train_model
from src.datasets import get_dataloaders, get_tokenizer
from src.models import GDConfig

if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    model = get_model_from_args()
    checkpoint = load_checkpoint(model)
    
    device = None
    flags = get_flags_from_args()
    if 'cpu' in flags:
        device = torch.device('cpu')
    
    tokenizer = get_tokenizer()
    model.resize_vocabulary(len(tokenizer))
    
    if model.config.train_dataset == 'tiny_stories':
        dataset_list = ['tiny_stories']
    elif model.config.train_dataset == 'children_stories':
        dataset_list = ['children_stories']
    else:
        dataset_list = ['tiny_stories', 'children_stories']
    
    dataloaders = get_dataloaders(d_seq=model.config.d_seq, tokenizer=tokenizer, batch_size=64, dataset_list=dataset_list)
    train_model(model, tokenizer, dataloaders, checkpoint=checkpoint, device=device)
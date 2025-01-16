import setup_paths
import torch
from script_util import get_model_from_args, get_tokenizer_and_dataset_from_args, load_checkpoint
from src.training import train_model
from src.datasets import get_dataloaders

if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    model = get_model_from_args()
    checkpoint = load_checkpoint(model)
    
    dataloaders = get_dataloaders(d_seq=model.config.d_seq)
    train_model(model, dataloaders, checkpoint=checkpoint)
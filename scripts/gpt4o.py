import setup_paths
import torch
from script_util import get_model_from_args, load_checkpoint, get_flags_from_args
from src.datasets import get_dataloaders, get_tokenizer
from src.evaluation import evaluate_model
from src.evaluation import generate_gpt4o_inputs, create_batch, check_batch, cancel_batch, parse_batch

if __name__ == "__main__":
  
    torch.manual_seed(0)
    
    flags = get_flags_from_args()
    
    if 'input' in flags:
        print("Creating inputs")
        model = get_model_from_args()
        checkpoint = load_checkpoint(model)
        
        device = None
        flags = get_flags_from_args()
        if 'cpu' in flags:
            device = torch.device('cpu')
        
        tokenizer = get_tokenizer()
        model.resize_vocabulary(len(tokenizer))
        
        dataloaders = get_dataloaders(d_seq=model.config.d_seq, tokenizer=tokenizer, batch_size=64)
            
    elif 'batch' in flags:
        print("Creating batch")
        create_batch()
    elif 'check' in flags:
        check_batch()
    elif 'parse' in flags:
        parse_batch()
    else:
        raise ValueError("No valid flags detected. Please use --input or --batch")
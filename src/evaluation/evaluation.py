import os
from tqdm import tqdm
import torch
from src.util import get_device

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/checkpoints')

def generate_ending(model, sequence, use_beam=True):
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        input = sequence.to(model.device)
    
        if use_beam:
            output = model.beam_search(input, max_new_tokens=100, num_beams=3, eos_token_id=model.eos_token_id)
        else:
            output = model.generate(input, max_new_tokens=100, eos_token_id=model.eos_token_id)
        
        return output[0].tolist()
    
def evaluate_model(model, tokenizer, dataloaders, checkpoint=None, num_generations=20, device=None):
    
    # Setup
    if device is None:
        device = get_device()
        
    model.device = device
    model.to(device)
    model.eos_token_id = tokenizer.eos_token_id
    
    if checkpoint is None:
        raise ValueError('No checkpoint provided, aborting evaluation')
        
    model.load_state_dict(checkpoint['model_state_dicts'][checkpoint['epoch']])
    
    test_dataset = dataloaders['test']
    
    print(f'Evaluating model [{model.config.get_name()}] with [{checkpoint["epoch"]}] on device [{model.device}]')
    
    num_sequence = 0
    num_failed = 0
    
    for batch in test_dataset:
        for i in range(batch['input_ids'].size(0)):
            sequence = batch['input_ids'][i, :].tolist()
            
            if tokenizer.pad_token_id in sequence:
                sequence = sequence[:sequence.index(tokenizer.pad_token_id)]
            
            sequence_length = len(sequence)
            if sequence_length < 2:
                num_failed += 1
                continue
            
            true_beginning_ids = sequence[:sequence_length // 2]
            true_ending_ids = sequence[sequence_length // 2:]
            generated_ending_ids = generate_ending(model, true_beginning_ids)
            
            true_beginning_text = tokenizer.decode(true_beginning_ids)
            true_ending_text = tokenizer.decode(true_ending_ids)
            generated_ending_text = tokenizer.decode(generated_ending_ids)
            
            print('*' * 80)
            print(f'True Story: {true_beginning_text} [{true_ending_text}]')
            print('=' * 80)
            print(f'Generated Ending: {true_beginning_text} [{generated_ending_text}]')
            
            num_sequence += 1
            if num_sequence >= num_generations:
                break
        
        if num_sequence >= num_generations:
                break
    
    print(f'Generated {num_sequence} sequences ({num_failed} failed)')
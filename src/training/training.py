import os
import time
from tqdm import tqdm
import torch
from torch.nn import functional as F
from src.util import get_device, get_time_remaining_formatted

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/checkpoints')

def model_forward(model, batch):
    sequence = batch.to(model.device)
    input_ids = sequence[:, :-1]
    target_ids = sequence[:, 1:]
    _, loss = model(input_ids, target_ids)
    return loss

def train_model(model, dataloaders, checkpoint=None, max_epochs=None):
    
    # Setup
    model.device = get_device()
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dicts'][checkpoint['epoch']])
    else:
        checkpoint = {
            'model_state_dicts': [],
            'train_losses': [],
            'valid_losses': [],
            'epoch': 0,
            'epoch_len': len(dataloaders['train'])
        }
    
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']
    
    num_steps_record = len(train_dataloader) // 1000 # Only saves model losses every 1000 steps for efficiency and memory reasons
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training Loop
    print(f'Training model [{model.get_name()}] on device [{model.device}]')
    
    while True:
        
        if max_epochs is not None and checkpoint['epoch'] >= max_epochs:
            print(f'Maximum number of epochs reached ({max_epochs}), stopping training')
            break
        
        train_loss = 0.0
        valid_loss = 0.0
        
        start_time = time.time()
        
        with tqdm(total=len(train_dataloader), desc=f'Epoch {checkpoint['epoch']}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                
                model.train()
                model.optimizer.zero_grad()
                
                train_loss = model_forward(model, batch)
                train_loss.backward()
                train_loss = train_loss.item()
                optimizer.step()
                
                if batch_idx % num_steps_record == 0:
                    model.eval()
                    valid_loss = 0.0
                    with torch.no_grad():
                        for valid_batch in valid_dataloader:
                            valid_loss += model_forward(model, valid_batch).item()
                    valid_loss /= len(valid_dataloader)
                    
                    current_step = epoch * len(train_dataloader) + batch_idx
                    checkpoint['train_losses'].append((current_step, round(train_loss, 4)))
                    checkpoint['valid_losses'].append((current_step, round(valid_loss, 4)))
                
                time_remaining = get_time_remaining_formatted(start_time, batch_idx, len(train_dataloader))
                pbar.set_postfix_str(f'Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Time Remaining: {time_remaining}')
                pbar.update(1)
        
        checkpoint['model_state_dicts'].append(model.state_dict())
        torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, f'{model.get_name()}_checkpoint.pth'))
        checkpoint['epoch'] += 1
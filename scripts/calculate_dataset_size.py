import setup_paths

from src.datasets import get_dataloaders, get_tokenizer

tokenizer = get_tokenizer()
dataloaders = get_dataloaders(d_seq=512, tokenizer=tokenizer, batch_size=64)

train_dataloader = dataloaders['train']

total_tokens_including_padding = 0
total_tokens_excluding_padding = 0

for i, batch in enumerate(train_dataloader):
    tokens = batch['input_ids']
    total_tokens_including_padding += tokens.numel()
    total_tokens_excluding_padding += tokens[tokens != tokenizer.pad_token_id].numel()
    
print(f'Total tokens including padding: {total_tokens_including_padding}')
print(f'Total tokens excluding padding: {total_tokens_excluding_padding}')
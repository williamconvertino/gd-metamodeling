import setup_paths
import torch
import torch.nn.functional as F
from script_util import get_model_from_args, load_checkpoint, get_flags_from_args
from src.datasets import get_dataloaders, get_tokenizer
from src.evaluation import evaluate_model

if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    model = get_model_from_args()
    checkpoint = load_checkpoint(model)
    
    device = None
    flags = get_flags_from_args()
    if 'cpu' in flags:
        device = torch.device('cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    wte = model.wte.weight
    
    mean_embedding_vector = wte.mean(dim=0)
    mean_embedding_norm = mean_embedding_vector.norm()

    embedding_norms = wte.norm(dim=1)

    mean_norm = embedding_norms.mean()
    std_norm = embedding_norms.std()

    z_score = mean_embedding_norm.item() / std_norm.item()

    relative_magnitude = mean_embedding_norm.item() / mean_norm.item()

    print(f"Mean Embedding Norm: {mean_embedding_norm.item():.6f}")
    print(f"Mean of Individual Embedding Norms: {mean_norm.item():.6f}")
    print(f"Std of Individual Embedding Norms: {std_norm.item():.6f}")
    print(f"Z-Score: {z_score:.6f}")
    print(f"Relative Magnitude: {relative_magnitude:.6f}")

    cosine_similarities = F.cosine_similarity(wte, mean_embedding_vector.unsqueeze(0), dim=1)
    mean_cosine_similarity = cosine_similarities.mean().item()

    print(f"Mean Cosine Similarity to Mean Vector: {mean_cosine_similarity:.6f}")
import torch


def calculate_sparsity(model):
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum(torch.sum(p == 0) for p in model.parameters())
    sparsity = zero_params / total_params
    return sparsity.item()

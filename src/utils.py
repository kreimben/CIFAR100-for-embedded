import copy

import torch


def calculate_sparsity(model):
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum(torch.sum(p == 0) for p in model.parameters())
    sparsity = zero_params / total_params
    return sparsity.item()


def get_quantized_model_from_weight(model, ckpt='quantized_resnet_cifar100.pth'):
    new_model = copy.deepcopy(model).cpu().eval()
    new_model.load_state_dict(torch.load(ckpt, map_location="cpu")['state_dict'])
    quantized_model = torch.quantization.convert(new_model, inplace=False)
    return quantized_model

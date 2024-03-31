import torch
from torch import nn
from torchvision import models

from eval import evaluate, dataloader
from src.utils import calculate_sparsity

pruned_model_dict = torch.load("pruned_model.pth", map_location=torch.device('cpu'))
pruned_model = models.resnet50(weights=None)
pruned_model.fc = nn.Linear(pruned_model.fc.in_features, 100)
pruned_model.load_state_dict(pruned_model_dict)
pruned_model.eval()  # Set the model to inference mode
print(f'Sparsity of pruned model is: {calculate_sparsity(pruned_model) * 100:.3f}%')

evaluate(pruned_model, dataloader, 'pruned model')

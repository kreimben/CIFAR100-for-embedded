import torch
from torch import nn
from torchvision import models

from eval import evaluate, dataloader

model_dict = torch.load("resnet_cifar100.pth", map_location=torch.device('cpu'))
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)
model.load_state_dict(model_dict)
model.eval()  # Set the model to inference mode

evaluate(model, dataloader, 'normal model')

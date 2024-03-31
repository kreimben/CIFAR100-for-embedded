import torch
from torch import nn
from torchvision import models

from eval import evaluate, dataloader

quantized_model = models.resnet50(weights=None)
quantized_model.fc = nn.Linear(quantized_model.fc.in_features, 100)
quantized_model.eval()  # Set the model to inference mode
quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
quantized_model = torch.quantization.prepare(quantized_model, inplace=True)


def calibrate_quantization(model, test_dataloader):
    # Run calibration
    i = 0
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            model(inputs)
            print(f'[Calibrating] {i + 1}  / {len(test_dataloader)} done.', end='\r')
            i += 1


calibrate_quantization(quantized_model, dataloader)
evaluate(quantized_model, dataloader, 'quantized model')

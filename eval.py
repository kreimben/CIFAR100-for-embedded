import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from src.utils import calculate_sparsity

model_dict = torch.load("resnet_cifar100.pth", map_location=torch.device('cpu'))
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)
model.load_state_dict(model_dict)
model.eval()  # Set the model to inference mode

pruned_model_dict = torch.load("pruned_model.pth", map_location=torch.device('cpu'))
pruned_model = models.resnet50(weights=None)
pruned_model.fc = nn.Linear(pruned_model.fc.in_features, 100)
pruned_model.load_state_dict(pruned_model_dict)
pruned_model.eval()  # Set the model to inference mode
print(f'Sparsity of pruned model is: {calculate_sparsity(pruned_model) * 100:.3f}%')

quantized_model = models.resnet50(weights=None)
quantized_model.fc = nn.Linear(quantized_model.fc.in_features, 100)
quantized_model.eval()  # Set the model to inference mode
quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
quantized_model = torch.quantization.prepare(model, inplace=True)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
dataset = torch.utils.data.random_split(dataset, [.99, .01])[1]
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


def evaluate(model, test_dataloader, model_name):
    total_time = 0
    total_batches = 0
    correct_predictions = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            x, y = batch
            start_time = time.time()

            pred_y = model(x)  # Forward pass

            correct_predictions += (pred_y.argmax(1) == y).type(torch.float).sum().item()

            total_time += (time.time() - start_time) * 1000
            total_batches += 1
            print(f'{i + 1} / {len(test_dataloader)} done.', end='\r')

        accuracy = correct_predictions / len(test_dataloader.dataset)

        print(
            f"Average Inference Time of {model_name} per 1 dataset: {total_time / total_batches:.3f} ms, "
            f"{model_name.capitalize()} accuracy: {accuracy * 100:.3f}%."
        )


def calibrate_quantization(model, test_dataloader):
    # Run calibration
    i = 0
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            model(inputs)
            print(f'[Calibrating] {i + 1}  / {len(test_dataloader)} done.', end='\r')
            i += 1


evaluate(model, dataloader, 'normal model')
evaluate(pruned_model, dataloader, 'pruned model')
calibrate_quantization(quantized_model, dataloader)
evaluate(quantized_model, dataloader, 'quantized model')

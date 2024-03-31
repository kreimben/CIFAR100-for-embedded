import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

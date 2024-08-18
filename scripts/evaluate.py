import torch
import numpy as np

def validate(model, test_loader):
    model.eval()
    confusion_matrix = np.zeros((10, 10), dtype=int)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(torch.device('cpu')), labels.to(torch.device('cpu'))
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for label, prediction in zip(labels, predicted):
                confusion_matrix[label.item(), prediction.item()] += 1

    return confusion_matrix


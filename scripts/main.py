import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_data_loaders
from src.cnn_model import CNN
from src.train import train
from src.evaluate import validate
from src.metrics import plot_confusion_matrix, calculate_metrics, plot_metrics

def main():
    # Get data loaders
    train_loader, test_loader = get_data_loaders()

    # Initialize model, loss function, and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    train_losses = train(model, train_loader, criterion, optimizer, num_epochs)

    # Validate the model
    confusion_matrix = validate(model, test_loader)
    plot_confusion_matrix(confusion_matrix)

    # Calculate and plot statistics
    precision, recall, f1_score = calculate_metrics(confusion_matrix)
    plot_metrics(precision, recall, f1_score)

if __name__ == "__main__":
    main()


# scripts/train.py

import torch
import torch.nn as nn
import argparse
from torch.optim import AdamW
from octmnist_classifier.model import SimpleCNN
from octmnist_classifier.preprocess import load_octmnist, balance_data, prepare_dataloaders
import os

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs, model_name):
    model.to(device)
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f"saved_model/{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"✅ New best model saved as: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCTMNIST CNN model with a specific sampling strategy")
    parser.add_argument("--strategy", type=str, choices=["smote", "smote_tomek", "undersample"], default="smote_tomek")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"model_{args.strategy}"

    print(f"📥 Loading OCTMNIST data...")
    X, y = load_octmnist(flatten=False)
    X, y = balance_data(X, y, strategy=args.strategy)

    print("🧹 Preparing dataloaders...")
    train_loader, val_loader, _ = prepare_dataloaders(X, y, batch_size=args.batch_size)

    print("🧠 Training model...")
    model = SimpleCNN()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, args.epochs, model_name)

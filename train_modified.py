
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from model import rPPGEstimatorCNN
from dataloader import PurePPGRPPGDataset

def train():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    writer = SummaryWriter(log_dir=config["log_dir"])

    # Dataset and loader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = PurePPGRPPGDataset(config["train_video_dir"], transform=transform)
    val_dataset = PurePPGRPPGDataset(config["val_video_dir"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize model
    model = rPPGEstimatorCNN(output_length=config["output_size"]).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % config["log_interval"] == 0:
                avg_loss = running_loss / config["log_interval"]
                print(f"Epoch [{epoch+1}/{config['epochs']}], Step [{batch_idx+1}], Loss: {avg_loss:.4f}")
                writer.add_scalar("Train/Loss", avg_loss, epoch * len(train_loader) + batch_idx)
                running_loss = 0.0

        # Validation after each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()

        val_loss /= len(val_loader)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        print(f"Epoch [{epoch+1}] Validation Loss: {val_loss:.4f}")

        # Save latest model checkpoint for resuming
        latest_ckpt_path = os.path.join(config["checkpoint_dir"], "latest_checkpoint.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, latest_ckpt_path)

        # Save best model checkpoint with epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config["checkpoint_dir"], f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, best_model_path)
            print(f"Best model saved at epoch {epoch+1} with loss {val_loss:.4f}.")

    writer.close()

if __name__ == "__main__":
    train()


import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from signal_model import SignalEstimatorNet
from signal_dataloader import RPPGToPhysioDataset

def train():
    with open("signal_config.json", "r") as f:
        config = json.load(f)

    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(config["log_dir"])

    # Dataset
    train_dataset = RPPGToPhysioDataset(config["train_dir"])
    val_dataset = RPPGToPhysioDataset(config["val_dir"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model
    model = SignalEstimatorNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.L1Loss()

    start_epoch = 0
    best_loss = float("inf")
    latest_ckpt_path = os.path.join(config["checkpoint_dir"], "latest_checkpoint.pth")

    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["val_loss"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = 0.0
        for rppg, ppg, ecg in train_loader:
            rppg, ppg, ecg = rppg.cuda(), ppg.cuda(), ecg.cuda()
            optimizer.zero_grad()
            ppg_pred, ecg_pred = model(rppg)
            loss = criterion(ppg_pred, ppg) + criterion(ecg_pred, ecg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Train/TotalLoss", avg_train_loss, epoch)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rppg, ppg, ecg in val_loader:
                rppg, ppg, ecg = rppg.cuda(), ppg.cuda(), ecg.cuda()
                ppg_pred, ecg_pred = model(rppg)
                loss = criterion(ppg_pred, ppg) + criterion(ecg_pred, ecg)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Val/TotalLoss", avg_val_loss, epoch)
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoints
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss
        }, latest_ckpt_path)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(config["checkpoint_dir"], f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss
            }, best_model_path)
            print(f"Saved best model at epoch {epoch+1} with loss {best_loss:.4f}")

    writer.close()

if __name__ == "__main__":
    train()

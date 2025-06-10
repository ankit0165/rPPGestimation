
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from signal_model import SignalEstimatorNet
from signal_dataloader import RPPGToPhysioDataset

def evaluate(model, dataloader, device):
    model.eval()
    all_ppg_preds, all_ecg_preds = [], []
    all_ppg_true, all_ecg_true = [], []

    with torch.no_grad():
        for rppg, ppg, ecg in dataloader:
            rppg = rppg.to(device)
            ppg = ppg.to(device)
            ecg = ecg.to(device)

            ppg_pred, ecg_pred = model(rppg)
            all_ppg_preds.append(ppg_pred.cpu().numpy())
            all_ecg_preds.append(ecg_pred.cpu().numpy())
            all_ppg_true.append(ppg.cpu().numpy())
            all_ecg_true.append(ecg.cpu().numpy())

    ppg_preds = np.concatenate(all_ppg_preds)
    ecg_preds = np.concatenate(all_ecg_preds)
    ppg_true = np.concatenate(all_ppg_true)
    ecg_true = np.concatenate(all_ecg_true)

    return ppg_preds, ecg_preds, ppg_true, ecg_true

def compute_metrics(preds, gts, name="PPG"):
    mae = mean_absolute_error(gts, preds)
    rmse = mean_squared_error(gts, preds, squared=False)
    print(f"{name} MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

def plot_signals(pred, gt, name="PPG"):
    plt.figure(figsize=(10, 3))
    plt.plot(gt[:300], label="Ground Truth")
    plt.plot(pred[:300], label="Prediction", linestyle="--")
    plt.title(f"{name} Signal Prediction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_prediction.png", dpi=300)
    plt.close()

def plot_bland_altman(pred, gt, name="PPG"):
    mean = (pred + gt) / 2
    diff = pred - gt
    md = np.mean(diff)
    sd = np.std(diff)

    plt.figure(figsize=(6, 4))
    plt.scatter(mean[:500], diff[:500], alpha=0.5)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--')
    plt.title(f"Bland-Altman Plot ({name})")
    plt.xlabel("Mean")
    plt.ylabel("Difference")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_bland_altman.png", dpi=300)
    plt.close()

def main():
    with open("signal_config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = RPPGToPhysioDataset(config["val_dir"])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = SignalEstimatorNet().to(device)
    best_model_path = sorted([f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("best_model")])[-1]
    checkpoint = torch.load(os.path.join(config["checkpoint_dir"], best_model_path))
    model.load_state_dict(checkpoint["model_state_dict"])

    ppg_preds, ecg_preds, ppg_true, ecg_true = evaluate(model, val_loader, device)

    compute_metrics(ppg_preds.flatten(), ppg_true.flatten(), name="PPG")
    compute_metrics(ecg_preds.flatten(), ecg_true.flatten(), name="ECG")

    plot_signals(ppg_preds[0], ppg_true[0], name="PPG")
    plot_signals(ecg_preds[0], ecg_true[0], name="ECG")
    plot_bland_altman(ppg_preds.flatten(), ppg_true.flatten(), name="PPG")
    plot_bland_altman(ecg_preds.flatten(), ecg_true.flatten(), name="ECG")

if __name__ == "__main__":
    main()

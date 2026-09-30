import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from deep_pomcp_env.nets.policy_value_net import DeepPOMCPNet


class TransitionDataset(Dataset):
    # Loads a single .npz dataset of Deep-POMCP transitions.

    def __init__(self, npz_path, label=""):
        print("  Loading: " + npz_path, flush=True)
        d = np.load(npz_path)
        self.particles      = torch.from_numpy(d["particles"])       # (N,200,4)
        self.kinematics     = torch.from_numpy(d["kinematics"])      # (N,8)
        self.local_grids    = torch.from_numpy(d["local_grids"])     # (N,121)
        self.policy_targets = torch.from_numpy(d["policy_targets"])  # (N,5)
        self.value_targets  = torch.from_numpy(d["value_targets"])   # (N,1)
        n = len(self.particles)
        wins = int((d["value_targets"] > 0).sum())
        print("  " + label + ": " + str(n) + " transitions | wins " + str(wins)
              + " (" + f"{100*wins/n:.1f}%" + ")", flush=True)

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, idx):
        return (
            self.particles[idx],
            self.kinematics[idx],
            self.local_grids[idx],
            self.policy_targets[idx],
            self.value_targets[idx]
        )


def subsample(dataset, n, rng):
    from torch.utils.data import Subset
    idxs = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False).tolist()
    return Subset(dataset, idxs)


def train(original_npz, adversarial_npz, output_weights,
          epochs=60, batch_size=512, val_fraction=0.15,
          alpha_value=1.0, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sep = "=" * 70
    print(sep, flush=True)
    print("  DEEP-POMCP RETRAINING (adversarial + original 50/50)", flush=True)
    print("  Device: " + str(device), flush=True)
    print(sep, flush=True)

    rng = np.random.default_rng(42)
    orig_ds = TransitionDataset(original_npz, label="Original")
    adv_ds  = TransitionDataset(adversarial_npz, label="Adversarial")

    target_n = min(len(orig_ds), len(adv_ds))
    orig_sub = subsample(orig_ds, target_n, rng)
    adv_sub  = subsample(adv_ds,  target_n, rng)
    combined = ConcatDataset([orig_sub, adv_sub])

    total_n = len(combined)
    n_val   = int(total_n * val_fraction)
    n_train = total_n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        combined, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print("  Mixed dataset:", flush=True)
    print("    Original subset:    " + str(len(orig_sub)), flush=True)
    print("    Adversarial subset: " + str(len(adv_sub)), flush=True)
    print("    Total:              " + str(total_n), flush=True)
    print("    Train / Val:        " + str(n_train) + " / " + str(n_val), flush=True)
    print(sep, flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    model     = DeepPOMCPNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    ce_fn  = nn.CrossEntropyLoss()
    mse_fn = nn.MSELoss()

    best_val  = float("inf")
    best_ep   = 0

    print("  " + "Epoch".ljust(6) + "TrainLoss".rjust(12) + "ValLoss".rjust(12)
          + "PolicyCE".rjust(12) + "ValueMSE".rjust(12) + "LR".rjust(12), flush=True)

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_sum, tr_steps = 0.0, 0
        for p, k, g, pt, vt in train_loader:
            p, k, g, pt, vt = (x.to(device) for x in (p, k, g, pt, vt))
            optimizer.zero_grad()
            pol, val = model(p, k, g)
            loss = ce_fn(pol, pt) + alpha_value * mse_fn(val, vt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_sum += loss.item(); tr_steps += 1

        avg_tr = tr_sum / max(tr_steps, 1)

        model.eval()
        vl_sum = vl_pce = vl_vmse = 0.0
        vl_steps = 0
        with torch.no_grad():
            for p, k, g, pt, vt in val_loader:
                p, k, g, pt, vt = (x.to(device) for x in (p, k, g, pt, vt))
                pol, val = model(p, k, g)
                lp = ce_fn(pol, pt); lv = mse_fn(val, vt)
                loss = lp + alpha_value * lv
                vl_sum += loss.item(); vl_pce += lp.item()
                vl_vmse += lv.item(); vl_steps += 1

        avg_vl   = vl_sum   / max(vl_steps, 1)
        avg_pce  = vl_pce   / max(vl_steps, 1)
        avg_vmse = vl_vmse  / max(vl_steps, 1)
        scheduler.step(avg_vl)
        cur_lr = optimizer.param_groups[0]["lr"]

        if avg_vl < best_val:
            best_val = avg_vl; best_ep = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": avg_vl, "val_accuracy": 100.0 * (1.0 - avg_vl)},
                       output_weights)

        if epoch % 5 == 0 or epoch == 1:
            mark = " *" if epoch == best_ep else ""
            print("  " + str(epoch).ljust(6)
                  + f"{avg_tr:12.4f}{avg_vl:12.4f}{avg_pce:12.4f}{avg_vmse:12.4f}{cur_lr:12.6f}"
                  + mark, flush=True)

    elapsed = time.time() - t0
    print(sep, flush=True)
    print("  TRAINING COMPLETE", flush=True)
    print("  Best epoch: " + str(best_ep) + " | Val loss: " + f"{best_val:.4f}", flush=True)
    print("  Time: " + f"{elapsed:.0f}s", flush=True)
    print("  Weights: " + output_weights, flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original",    type=str,
        default=r"d:\Btech Project\project\deep_pomcp_dataset.npz")
    parser.add_argument("--adversarial", type=str,
        default=os.path.join(current_dir, "adversarial_dataset.npz"))
    parser.add_argument("--output",      type=str,
        default=r"d:\Btech Project\project\deep_pomcp_weights_v2.pth")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch_size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--alpha_value", type=float, default=1.0)
    args = parser.parse_args()
    train(
        original_npz=args.original,
        adversarial_npz=args.adversarial,
        output_weights=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_value=args.alpha_value
    )

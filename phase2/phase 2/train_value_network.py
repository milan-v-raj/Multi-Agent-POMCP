import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- 1. DEFINE THE DATASET ---
class SwarmDataset(Dataset):
    def __init__(self, x_file, y_file):
        print("Loading data from disk...")
        self.X = np.load(x_file).astype(np.float32)
        self.Y = np.load(y_file).astype(np.float32).reshape(-1, 1)
        
        # Quick Bias Check
        wins = np.sum(self.Y > 0)
        losses = np.sum(self.Y < 0)
        print(f"Loaded {len(self.X)} states. Wins: {wins} | Losses: {losses}")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

# --- 2. THE UPGRADED DEEP NETWORK ---
class SwarmValueNet(nn.Module):
    def __init__(self):
        super(SwarmValueNet, self).__init__()
        # Wider layers, Batch Normalization to prevent gradient vanishing
        self.network = nn.Sequential(
            nn.Linear(22, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.network(x)

# --- 3. THE TRAINING LOOP ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = SwarmDataset("swarm_states_X.npy", "swarm_labels_Y.npy")
    # Increased batch size for smoother gradients on large datasets
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True) 

    model = SwarmValueNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    # NEW: Learning Rate Scheduler. If loss stops dropping, reduce the learning rate.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 150 # We can use fewer epochs because the network is smarter

    print("--- STARTING TRAINING ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss) # Tell the scheduler to check the loss
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "swarm_value_net.pth")
    print("--- TRAINING COMPLETE ---")
    print("Saved trained model to 'swarm_value_net.pth'")

if __name__ == "__main__":
    train_model()
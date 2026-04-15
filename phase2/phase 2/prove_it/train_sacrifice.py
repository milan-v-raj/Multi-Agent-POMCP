import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SwarmValueNet(nn.Module):
    def __init__(self):
        super(SwarmValueNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() 
        )
    def forward(self, x):
        return self.network(x)

def train():
    print("Training Map-Agnostic Relative 4D Node...")
    X, Y = [], []
    
    for _ in range(100000):
        # 1. Spawn drone anywhere
        x, y = np.random.uniform(0, 800), np.random.uniform(0, 600)
        
        # 2. Spawn its objective ANYWHERE (This makes it map-agnostic!)
        obj_x, obj_y = np.random.uniform(0, 800), np.random.uniform(0, 600)
        
        role = np.random.choice([0.0, 1.0])
        gate = np.random.choice([0.0, 1.0])
        
        # 3. Calculate distance to objective
        dist = np.linalg.norm([x - obj_x, y - obj_y])
        
        # 4. The Plateau: If it's within 30 pixels (a 60x60 box), perfect score!
        if dist < 30.0:
            dist = 0.0
            
        score = 1.0 - (dist / 1000.0)
        
        # 5. Calculate Relative Vectors (Normalized)
        dx = (obj_x - x) / 800.0
        dy = (obj_y - y) / 600.0

        X.append([dx, dy, role, gate]) 
        Y.append([score * 2.0 - 1.0]) 

    X_t = torch.FloatTensor(X)
    Y_t = torch.FloatTensor(Y)
    model = SwarmValueNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("--- STARTING TRAINING ---")
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, Y_t)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "swarm_value_net.pth")
    print("💾 Map-Agnostic 4D Node saved!")

if __name__ == "__main__":
    train()
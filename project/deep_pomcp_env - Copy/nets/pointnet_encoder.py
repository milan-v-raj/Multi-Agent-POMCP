import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, input_dim: int = 4, embedding_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        if particles.dim() == 2:
            particles = particles.unsqueeze(0)
        x = F.relu(self.ln1(self.fc1(particles)))
        x = F.relu(self.ln2(self.fc2(x)))
        global_feature, _ = torch.max(x, dim=1)
        return global_feature

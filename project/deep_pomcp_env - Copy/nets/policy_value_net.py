import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .pointnet_encoder import PointNetEncoder

class DeepPOMCPNet(nn.Module):
    def __init__(self, num_actions: int = 5, num_particles: int = 200, particle_dim: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.pointnet = PointNetEncoder(input_dim=particle_dim, embedding_dim=128)
        self.grid_encoder = nn.Sequential(
            nn.Linear(121, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.kin_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        self.trunk = nn.Sequential(
            nn.Linear(224, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, particles: torch.Tensor, kinematics: torch.Tensor, local_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if kinematics.dim() == 1: kinematics = kinematics.unsqueeze(0)
        if local_grid.dim() == 1: local_grid = local_grid.unsqueeze(0)
        p_feat = self.pointnet(particles)
        g_feat = self.grid_encoder(local_grid)
        k_feat = self.kin_encoder(kinematics)
        fusion = torch.cat([p_feat, g_feat, k_feat], dim=-1)
        trunk_out = self.trunk(fusion)
        policy_logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)
        return policy_logits, value

    def predict_priors_and_value(self, particles: torch.Tensor, kinematics: torch.Tensor, local_grid: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.eval()
        with torch.no_grad():
            logits, val = self.forward(particles, kinematics, local_grid)
            priors = F.softmax(logits, dim=-1).squeeze(0)
            scalar_val = float(val.item())
        return priors, scalar_val

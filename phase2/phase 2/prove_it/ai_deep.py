import torch
import torch.nn as nn
import numpy as np
import pygame
from config import *

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

value_net = SwarmValueNet()
try:
    value_net.load_state_dict(torch.load("swarm_value_net.pth", map_location='cpu', weights_only=True))
    value_net.eval()
    print("🧠 Relative Map-Agnostic Network Loaded!")
except FileNotFoundError:
    print("⚠️ WARNING: swarm_value_net.pth not found!")

# 16-Direction Compass + Brakes
angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
ACTIONS = [np.array([0.0, 0.0])] + [np.array([np.cos(a), np.sin(a)]) for a in angles]

# THE FIX: Added switch_pos and gate_wait_pos to the function signature
def get_deep_action(hunter, ally_pos, target_guess, spread, arena, switch_pos, gate_wait_pos):
    
    switch_rect = pygame.Rect(switch_pos[0]-30, switch_pos[1]-30, 60, 60)
    gate_active = not (switch_rect.collidepoint(hunter.pos[0], hunter.pos[1]) or 
                       switch_rect.collidepoint(ally_pos[0], ally_pos[1]))
    gate_open_val = 0.0 if gate_active else 1.0

    # Role Assignment
    d_me = np.linalg.norm(hunter.pos - switch_pos)
    d_ally = np.linalg.norm(ally_pos - switch_pos)
    my_role = 0.0 if d_me <= d_ally else 1.0

    # --- DYNAMIC OBJECTIVE SELECTION ---
    if my_role == 0.0:
        obj_pos = switch_pos 
    else:
        if gate_open_val == 0.0:
            obj_pos = gate_wait_pos 
        else:
            obj_pos = target_guess 

    best_action = np.array([0.0, 0.0])
    best_val = -float('inf')

    for action in ACTIONS:
        sim_pos = hunter.pos + (action * 15.0)

        wall_penalty = 0.0
        if not arena.is_walkable(sim_pos, padding=0):
            wall_penalty = -2.0 

        # RELATIVE VECTORS
        dx = (obj_pos[0] - sim_pos[0]) / WIDTH
        dy = (obj_pos[1] - sim_pos[1]) / HEIGHT

        state_4d = [dx, dy, my_role, gate_open_val]
        tensor = torch.FloatTensor(state_4d).unsqueeze(0)
        
        with torch.no_grad():
            val = value_net(tensor).item()

        total_score = val + wall_penalty

        if total_score > best_val:
            best_val = total_score
            best_action = action

    return best_action
# Deep-POMCP: Scalable Neural Belief-Space Planning for Multi-Agent Pursuit-Evasion Under Continuous Partial Observability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**B.Tech Capstone Project**  
*Department of Electronics and Communication Engineering (ECE)*  
**National Institute of Technology Calicut (NITC)**  
*Authors: Angelia Aju Mathai, Milan V Raj, Naithen Sam*  
*Supervisor: Dr. Anup Aprem*

---

## 📌 Overview

Cooperative multi-agent pursuit-evasion under continuous dynamics and partial observability represents a canonical **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**. Conventional model-free Multi-Agent Reinforcement Learning (MARL) methods struggle with long-horizon non-greedy coordination and catastrophic belief collapse under line-of-sight occlusions. Conversely, classical Partially Observable Monte Carlo Planning (POMCP) suffers from rollout trajectory divergence, sample inefficiency, and high computational overhead (>150 ms latency).

**Deep-POMCP** is an online neural belief-space planner integrating:
1. **Permutation-Invariant PointNet Belief Encoder**: Operates directly over raw Sequential Monte Carlo (SMC) particle clouds ($\mathcal{B}_t \in \mathbb{R}^{200 \times 4}$), preserving multimodal bifurcations and reducing information loss ($D_{\text{KL}}$) by **46.6%** over Gaussian approximations.
2. **Depth-Truncated PUCT Tree Search ($D=6$)**: Replaces noisy Monte Carlo rollouts with dual policy-value neural guidance ($V_\phi, \mathbf{P}_\theta$), bounded by Bellman $\gamma$-contraction (Theorem 1), slashing decision latency from **189 ms down to 0.91 ms**.
3. **Internal Adversarial Simulation & Encirclement Advantage ($R_{\text{EA}}$)**: Embeds a 12-ray dynamic evader model into lookahead expansions paired with a $180^\circ$ pincer reward, boosting capture rate against active evasion to **70.0%** (where Reactive A* drops to 10.0%).
4. **Cooperative Sacrifice Resolution**: Solves multi-stage delayed gratification puzzles (Switch-Door) with **100.0% switch trigger** and **63.3% capture rate zero-shot** (where classical baselines achieve 0.0%).
5. **Information-Theoretic Event-Triggered MCTS (ET-MCTS)**: Triggers replanning only on significant entropy flux, line-of-sight edges, or path invalidation, pruning search invocations by **39.3%** and reducing decision latency to sub-millisecond regimes (**4.67 ms**).

---

## 🚀 Key Benchmark Results

| Benchmark Evaluation | Metric | Reactive A* | Vanilla POMCP | Heuristic POMCP | Deep-POMCP (Periodic) | ET-Deep-POMCP (Ours) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Unseen Maps ($N=80$)** | Win Rate (%) | 100.0% | 85.0% | 80.0% | **100.0%** | **100.0%** |
| | Wall Hits / Ep | 2.1 | 8.0 | 2.3 | **0.3** | **0.3** |
| | Latency (ms) | **0.17 ms** | 1.19 ms | 3.46 ms | 0.91 ms | **0.45 ms** |
| **Exp B: Speed Asymmetry ($2.0\times$)** | Win Rate (%) | 100.0% | 40.0% | 55.0% | **100.0%** | **100.0%** |
| **Exp C: Multimodal Belief** | $D_{\text{KL}}$ Loss (nats) | --- | --- | --- | **0.0541 nats** | **0.0541 nats** |
| | Info Preservation | Baseline | Baseline | Baseline | **+46.6% Gain** | **+46.6% Gain** |
| **Exp D: Strategic Adversary ($N=240$)**| Win Rate (%) | 10.0% | 40.0% | 60.0% | **70.0%** | 66.7% |
| | Mean Steps to Capture | 321.3 | 1007.2 | 738.9 | **622.5** | 681.4 |
| **Exp E: Cooperative Sacrifice ($N=120$)**| Win Rate (%) | **0.0%** | **0.0%** | **0.0%** | **63.3%** | **63.3%** |
| | Switch Trigger Rate | **0.0%** | **0.0%** | **0.0%** | **100.0%** | **100.0%** |
| **Cul-de-Sac Deadlock (`u_trap`)** | Win Rate (%) | 0.0% | 20.0% | 40.0% | 20.0% | **60.0%** |
| | Search Invocations | --- | --- | --- | 166.0 calls | **70.0 calls (-57.8%)** |

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/milan-v-raj/Multi-Agent-POMCP.git
   cd Multi-Agent-POMCP
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Interactive Live Showcases

All demonstrations run at 60 FPS in interactive Pygame environments:

### 1. Main Benchmark Showcase (Deep-POMCP vs. Baselines)
```bash
python project/deep_pomcp_env/demo_deep_pomcp.py
```
* **Controls**:
  * `[M]`: Cycle Pursuer AI (`DEEP-POMCP (Ours)` $\to$ `REACTIVE A*` $\to$ `HEURISTIC POMCP` $\to$ `VANILLA POMCP`)
  * `[E]`: Toggle Evader AI (`Strategic Adversarial` $\leftrightarrow$ `Standard Raycast`)
  * `[1 - 7]`: Switch Map Topology (1: Open, 2: Moderate, 3: Dense Maze, 4: Extreme Clutter, 5: U-Trap, 6: Figure-8, 7: Bimodal Fork)
  * `[SPACE]`: Pause / Resume
  * `[R]`: Reset with a new randomized seed
  * `[ESC]`: Exit

### 2. Cooperative Sacrifice Showcase (Switch-Door Puzzle)
```bash
python project/deep_pomcp_env/demo_switch_door.py
```
* Shows the multi-agent delayed gratification test where one pursuer must step onto and hold a floor switch to open a blast door for its teammate.

### 3. Event-Triggered MCTS Showcase (ET-MCTS)
```bash
python project/deep_pomcp_env/demo_et_pomcp.py
```
* Real-time HUD displaying dynamic event triggers (`[TRIGGER: ENTROPY FLUX]`, `[TRIGGER: LOS TRANSITION]`, `[TRIGGER: PATH INVALID]`, `[IDLE: A* TRACKING]`).

---

## 📂 Repository Structure

```
├── project/
│   ├── deep_pomcp_env/          # Core environment and algorithms
│   │   ├── baselines/           # Deep-POMCP, ET-MCTS, Reactive A*, POMCP
│   │   ├── nets/                # PointNet encoder & Policy-Value networks
│   │   ├── core_env.py          # Continuous Dec-POMDP simulator
│   │   ├── switch_door_env.py   # Cooperative sacrifice environment
│   │   ├── evaders.py           # Reactive and Strategic Adversarial evaders
│   │   ├── pathfinder.py        # Continuous A* trajectory generator
│   │   └── demos                # Interactive Pygame showcases
│   ├── deep_pomcp_weights.pth   # Pretrained neural network weights
│   └── benchmarking.py          # Experimental evaluation suites
├── midsem_report/               # IEEEtran conference paper source & figures
│   ├── midsem_report.tex        # Full LaTeX paper
│   └── figures/                 # Publication-quality vector diagrams & plots
├── requirements.txt             # Environment dependencies
├── WALKTHROUGH.md               # Detailed technical walkthrough
└── README.md                    # Project documentation
```

---

## 📜 Citation

If you use this work or codebase, please cite:
```bibtex
@inproceedings{mathai2026deeppomcp,
  title={Deep-POMCP: Scalable Neural Belief-Space Planning for Multi-Agent Pursuit-Evasion Under Continuous Partial Observability},
  author={Mathai, Angelia Aju and Raj, Milan V and Sam, Naithen and Aprem, Anup},
  booktitle={B.Tech Capstone Project, Department of ECE, NIT Calicut},
  year={2026}
}
```

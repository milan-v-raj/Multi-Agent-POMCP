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
4. **Macro-Action Dec-POMDP (Mac-POMCP) for Cooperative Sacrifice**: Resolves the horizon truncation trap in delayed-gratification switch-door puzzles with zero hardcoded coordinates, achieving **80.0%–100.0% win rate** where primitive flat POMCP and Greedy A* score **0.0%**.
5. **Epistemic Causal Mechanism Discovery (`causal_switch_pomdp`)**: Places $K=3$ unlabelled candidate pressure plates in the arena where only one latent switch $\kappa^* \in \{0, 1, 2\}$ physically operates the door. Agents maintain a dual belief (continuous evader particle cloud $\mathcal{B}_e$ + categorical causal belief $\mathbf{b}_\kappa \in \Delta^K$), prune non-functional switches via Bayesian negative evidence, and coordinate sacrifice (**100.0% Win Rate** for Causal Mac-POMCP; **90.0% Zero-Shot Win Rate** for frozen Deep Causal Mac-POMCP).
6. **Information-Theoretic Event-Triggered MCTS (ET-MCTS)**: Triggers replanning only on significant entropy flux, line-of-sight edges, or path invalidation, pruning search invocations by **39.3%** and reducing decision latency to sub-millisecond regimes (**4.67 ms**).

---

## 🚀 Key Benchmark Results

### 1. Core Pursuit-Evasion & ET-MCTS Benchmarks
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
| **Cul-de-Sac Deadlock (`u_trap`)** | Win Rate (%) | 0.0% | 20.0% | 40.0% | 20.0% | **60.0%** |
| | Search Invocations | --- | --- | --- | 166.0 calls | **70.0 calls (-57.8%)** |

### 2. Epistemic Causal Mechanism Discovery ($K=3$ Candidate Switches, Zero Hardcoding)
| Policy | Win Rate (Capture %) | Causal Discovery Rate | Mean Switches Tested | Mean Steps to Capture (TTC) |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy A\*** | 0.0% | 0.0% | 0.00 | Timeout (1200) |
| **Heuristic Trial-and-Error** | 0.0% | 100.0% | 2.20 | Timeout (1200) |
| **Causal Mac-POMCP (Ours)** | **100.0%** | **100.0%** | **2.00** | **649.4** |
| **Deep Causal Mac-POMCP (Frozen NN)** | **90.0%** | **100.0%** | **2.60** | **786.2** |

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

## 🎮 Interactive Showcases & Benchmarks

### 1. Main Benchmark Showcase (Deep-POMCP vs. Baselines)
```bash
python project/deep_pomcp_env/demo_deep_pomcp.py
```

### 2. Audited Single-Switch Cooperative Sacrifice Benchmark (`switch_door_pomdp`)
```bash
python project/switch_door_pomdp/run_benchmark.py --episodes 10
```

### 3. Epistemic Causal Mechanism Discovery Benchmark (`causal_switch_pomdp`)
```bash
python project/causal_switch_pomdp/run_benchmark.py --episodes 10 --switches 3 --output causal_benchmark_N10_v2.csv
```

### 4. Event-Triggered MCTS Showcase (ET-MCTS)
```bash
python project/deep_pomcp_env/demo_et_pomcp.py
```

---

## 📂 Repository Structure

```
├── project/
│   ├── causal_switch_pomdp/     # Epistemic Causal Discovery Dec-POMDP (K=3 latent switches)
│   ├── switch_door_pomdp/       # Audited Macro-Action Dec-POMDP (Single-switch sacrifice)
│   ├── deep_pomcp_env/          # Core environment, PointNet, AlphaZero PUCT, ET-MCTS
│   ├── deep_pomcp_weights.pth   # Pretrained frozen neural network weights
│   └── benchmarking.py          # Experimental evaluation suites
├── midsem_report/               # IEEEtran conference paper source & figures
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

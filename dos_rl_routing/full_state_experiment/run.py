"""Small, separate comparison of node-only and full-snapshot route keys.

Run from dos_rl_routing/:
    python -m full_state_experiment.run --seeds 3 --episodes 100 --test-episodes 30
Outputs only ``full_state_experiment/results.csv``. This is a smoke experiment,
not a closed-loop deployment test or evidence of statistical superiority.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.tabular import QLearningAgent
from dos_rl.config import NET_RL
from dos_rl.topology import select_sources
from dos_rl.trainer import evaluate, train
from full_state_experiment.full_state_mdp import FullStateRoutingMDP


class KeyCoverageAgent(QLearningAgent):
    """Track whether evaluation decisions use a Q-table state from training."""
    def begin_evaluation(self):
        self.training_keys = set(self.Q)
        self.eval_decisions = 0
        self.eval_seen = 0

    def greedy(self, state, actions):
        if hasattr(self, 'training_keys'):
            self.eval_decisions += 1
            self.eval_seen += int(state in self.training_keys)
        return super().greedy(state, actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--test-episodes', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.csv'))
    args = parser.parse_args()
    if min(args.seeds, args.episodes, args.test_episodes) < 1:
        parser.error('all counts must be positive')

    cfg = SimConfig()
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(3000 + seed)
        topo = deploy_topology(cfg.deployment, rng)
        sources = select_sources(topo, cfg.traffic.n_sources, rng)
        for mode in ('node', 'full_snapshot'):
            sim = NetworkSimulator(topo, cfg)
            if mode == 'node':
                mdp = RoutingMDP(sim, cfg, NET_RL, BeliefTracker(topo), state_mode='node')
            else:
                mdp = FullStateRoutingMDP(sim, cfg, NET_RL, BeliefTracker(topo))
            agent = KeyCoverageAgent(NET_RL, n_objectives=3)
            hist = train(agent, mdp, sources, episodes=args.episodes, seed=seed)
            n_keys = len(agent.Q)
            agent.begin_evaluation()
            test = evaluate(agent, mdp, sources, episodes=args.test_episodes, seed=9000 + seed)
            width = min(20, len(hist))
            row = {
                'state_mode': mode, 'seed': seed,
                'training_episodes': args.episodes, 'test_episodes': args.test_episodes,
                'early_train_return': np.mean([h.return_ for h in hist[:width]]),
                'late_train_return': np.mean([h.return_ for h in hist[-width:]]),
                'test_return': test['return_mean'], 'test_success_pct': test['success_rate'],
                'training_q_states': n_keys,
                'evaluation_decisions': agent.eval_decisions,
                'evaluation_decisions_with_seen_state': agent.eval_seen,
            }
            rows.append(row)
            print(f"seed {seed} {mode:13s} test return {row['test_return']:+.2f} "
                  f"success {row['test_success_pct']:.1f}% "
                  f"seen-state decisions {agent.eval_seen}/{agent.eval_decisions}")
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved separate results to {path}')


if __name__ == '__main__':
    main()

"""Regression and integration tests for the isolated full-snapshot MDP.

Run: python -m unittest full_state_experiment.test_full_state -v
"""
import copy
import unittest

import numpy as np

from dos_rl import BeliefTracker, NetworkSimulator, SimConfig, deploy_topology
from dos_rl.agents.tabular import QLearningAgent
from dos_rl.config import NET_RL
from dos_rl.trainer import evaluate, train
from full_state_experiment.full_state_mdp import FullRouteState, FullStateRoutingMDP


class FullStateTests(unittest.TestCase):
    def setUp(self):
        cfg = SimConfig()
        topo = deploy_topology(cfg.deployment, np.random.default_rng(3000))
        self.mdp = FullStateRoutingMDP(NetworkSimulator(topo, cfg), cfg, NET_RL, BeliefTracker(topo))
        # Start from a reachable multi-hop relay with a real neighbour action.
        self.source = next(i for i in range(topo.n_nodes)
                           if topo.dist_bs[i] > cfg.deployment.comm_range_m
                           and any(j >= 0 for j in self.mdp.actions(i)))
        self.mdp.sample_episode(self.source, np.random.default_rng(10), energy_spread=0.05)
        self.mdp.reset(self.source)
        self.action = next(a for a in self.mdp.actions(self.source) if a >= 0)

    def test_relevant_context_changes_state_and_snapshot_is_immutable(self):
        baseline = self.mdp.encode(self.source)
        self.assertIsInstance(baseline, FullRouteState)
        self.assertEqual(hash(baseline), hash(baseline))
        i, j = self.source, self.action
        changes = [
            lambda m: m.ep.belief.energy.__setitem__(j, 0.123),
            lambda m: m.ep.belief.queue_frac.__setitem__(j, 0.789),
            lambda m: m.ep.belief.load.__setitem__(i, 0.234),
            lambda m: m.ep.belief.prr.__setitem__((i, j), 0.345),
            lambda m: m.ep.eligible.__setitem__(j, not m.ep.eligible[j]),
            lambda m: m.ep.hops.__setitem__(j, m.ep.hops[j] + 1),
            lambda m: m.sim.link_up.__setitem__((i, j), not m.sim.link_up[i, j]),
            lambda m: m._visited.add(j),
            lambda m: setattr(m, '_t', m._t + 1),
        ]
        for mutate in changes:
            model = copy.deepcopy(self.mdp)
            mutate(model)
            self.assertNotEqual(baseline, model.encode(i))
        self.mdp.ep.belief.energy[j] = 0.01
        self.assertNotEqual(baseline, self.mdp.encode(i))
        # The key created before mutation is still identical to a clean copy.
        self.assertNotEqual(baseline.energy, self.mdp.encode(i).energy)

    def test_equal_state_same_action_and_random_draw_have_same_transition(self):
        for seed in range(12):
            left, right = copy.deepcopy(self.mdp), copy.deepcopy(self.mdp)
            state = left.encode(self.source)
            self.assertEqual(state, right.encode(self.source))
            self.assertEqual(left.actions(self.source), right.actions(self.source))
            np.testing.assert_array_equal(left.rewards(self.source, self.action),
                                          right.rewards(self.source, self.action))
            a = left.step(self.action, np.random.default_rng(seed))
            b = right.step(self.action, np.random.default_rng(seed))
            self.assertEqual((a.next_state, a.next_node, a.done, a.terminal, a.info),
                             (b.next_state, b.next_node, b.done, b.terminal, b.info))
            np.testing.assert_array_equal(a.rewards, b.rewards)
            self.assertEqual(a.scalar_reward, b.scalar_reward)
            self.assertEqual(a.shaping, b.shaping)

    def test_failure_terminal_is_separate_absorbing_key(self):
        class FailedDraw:
            def random(self):
                return 1.0
        res = self.mdp.step(self.action, FailedDraw())
        self.assertTrue(res.done)
        self.assertEqual(res.next_state, ('terminal', res.terminal))
        self.assertEqual(res.terminal, 'dead_end')

    def test_requires_active_current_relay_and_only_full_mode(self):
        with self.assertRaises(ValueError):
            self.mdp.encode((self.source + 1) % self.mdp.n)
        with self.assertRaises(ValueError):
            self.mdp.encode(self.source, mode='node')

    def test_existing_q_learning_trainer_accepts_full_state(self):
        agent = QLearningAgent(NET_RL, n_objectives=3)
        history = train(agent, self.mdp, [self.source], episodes=20, seed=19, all_nodes=False)
        self.assertEqual(len(history), 20)
        self.assertGreater(agent.q_size(), 0)
        result = evaluate(agent, self.mdp, [self.source], episodes=3, seed=1001)
        self.assertEqual(result['n_episodes'], 3)
        self.assertTrue(np.isfinite(result['return_mean']))


if __name__ == '__main__':
    unittest.main()

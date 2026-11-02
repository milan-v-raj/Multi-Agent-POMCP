"""Regression checks for corrected network reward and selection updates."""
import unittest
import numpy as np

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.tabular import QLearningAgent, SarsaAgent
from dos_rl.agents.linear_fa import LinearQAgent
from dos_rl.config import NET_RL
from dos_rl.trainer import train


class ImplementationAuditTests(unittest.TestCase):
    def test_network_single_objective_uses_weighted_reward(self):
        rl = NET_RL.with_(reward_weights=(1.0, 2.0, 1.0))
        for cls in (QLearningAgent, SarsaAgent):
            agent = cls(rl, n_objectives=1)
            agent.update_transition((0,), 1, np.array([0.1, 0.9, 0.5]),
                                    None, None, True, 0.0)
            self.assertAlmostEqual(agent.q((0,), [1])[0, 0], 0.3 * 0.6)

    def test_vector_updates_keep_objectives_separate(self):
        agent = QLearningAgent(NET_RL, n_objectives=3)
        agent.update_transition((0,), 1, np.array([0.1, 0.9, 0.9]), None, (), True, 0.0)
        np.testing.assert_allclose(agent.q((0,), [1])[0], [0.03, 0.27, 0.27])

    def test_reject_bad_weight_shape(self):
        agent = QLearningAgent(NET_RL.with_(reward_weights=(1.0, 1.0)), n_objectives=1)
        with self.assertRaises(ValueError):
            agent.update_transition((0,), 1, np.array([0.1, 0.9, 0.9]), None, (), True, 0.0)

    def test_default_dos_selection_ignores_tracked_variance(self):
        agent = QLearningAgent(NET_RL, n_objectives=3)
        state, actions = (0,), [1, 2]
        agent.q(state, actions)
        agent.var(state, actions)
        agent.Q[state][1][:] = [1.0, 4.0, 2.0]
        agent.Q[state][2][:] = [2.0, 1.0, 3.0]
        first = agent.select_objective(state, actions)
        agent.VAR[state][1][:] = [0.0, 100.0, 0.0]
        agent.VAR[state][2][:] = [0.0, 100.0, 0.0]
        self.assertEqual(first, 1)
        self.assertEqual(agent.select_objective(state, actions), first)
        linear = LinearQAgent(NET_RL)
        self.assertEqual(linear.select_objective(agent.q(state, actions), agent.var(state, actions)), 1)

    def test_linear_update_bootstraps_each_objective(self):
        agent = LinearQAgent(NET_RL, n_features=1, n_objectives=3)
        agent.theta[:, 0] = [1.0, 2.0, 3.0]
        agent.update(np.array([1.0]), (0,), 0, 1, np.zeros(3),
                     np.array([[2.0]]), False, 0.0)
        np.testing.assert_allclose(agent.theta[:, 0], [1.21, 2.42, 3.63])

    def test_gateway_features_do_not_index_last_relay(self):
        cfg = SimConfig()
        topo = deploy_topology(cfg.deployment, np.random.default_rng(3000))
        tracker = BeliefTracker(topo)
        tracker.reset()
        tracker.state.energy[-1] = 0.2
        feats = tracker.state.node_features(0, -1, gateway_prr=0.8)
        self.assertEqual(feats[1], 1.0)
        self.assertEqual(feats[3], 0.0)
        self.assertEqual(feats[6], 0.8)
        self.assertEqual(feats[7], 1.0)

    def test_terminal_shaping_uses_zero_terminal_potential(self):
        cfg = SimConfig()
        topo = deploy_topology(cfg.deployment, np.random.default_rng(3000))
        mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, NET_RL, BeliefTracker(topo))
        source = next(i for i in range(topo.n_nodes)
                      if topo.dist_bs[i] <= cfg.deployment.comm_range_m)
        mdp.sample_episode(source, np.random.default_rng(42))
        mdp.reset(source)
        result = mdp.step(-1, np.random.default_rng(1))
        self.assertTrue(result.done)
        self.assertAlmostEqual(result.shaping, -mdp.potential(source))

    def test_network_shaping_changes_q_learning_and_sarsa_updates(self):
        cfg = SimConfig()
        topo = deploy_topology(cfg.deployment, np.random.default_rng(3000))
        source = next(i for i in range(topo.n_nodes)
                      if topo.dist_bs[i] > cfg.deployment.comm_range_m)
        for cls in (QLearningAgent, SarsaAgent):
            agents = []
            for use_shaping in (False, True):
                rl = NET_RL.with_(use_shaping=use_shaping)
                mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
                agent = cls(rl, n_objectives=3)
                train(agent, mdp, [source], episodes=1, seed=42, all_nodes=False)
                agents.append(agent)
            common = [(state, action) for state in agents[0].Q.keys() & agents[1].Q.keys()
                      for action in agents[0].Q[state].keys() & agents[1].Q[state].keys()]
            self.assertTrue(any(not np.allclose(agents[0].Q[s][a], agents[1].Q[s][a])
                                for s, a in common), cls.__name__)


if __name__ == '__main__':
    unittest.main()

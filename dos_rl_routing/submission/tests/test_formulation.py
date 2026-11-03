"""Formulation checks and documented approximation boundaries.

Run with ``python -m unittest tests.test_formulation -v``.
These tests characterise the current model; they do not assert that its
compact tabular representation is a sufficient Markov state.
"""
import unittest
import numpy as np

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.config import NET_RL
from dos_rl.observer import ControllerObserver
from dos_rl.topology import select_sources


class FormulationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimConfig()
        rng = np.random.default_rng(3000)
        self.topo = deploy_topology(self.cfg.deployment, rng)
        self.sources = select_sources(self.topo, self.cfg.traffic.n_sources, rng)
        self.sim = NetworkSimulator(self.topo, self.cfg)

    def mdp(self, mode="node"):
        return RoutingMDP(self.sim, self.cfg, NET_RL, BeliefTracker(self.topo), state_mode=mode)

    def test_same_local_key_can_alias_different_neighbour_rewards(self):
        for mode in ("node", "belief"):
            mdp = self.mdp(mode)
            i = self.sources[0]
            j = next(a for a in mdp.actions(i) if a >= 0)
            key = mdp.encode(i)
            before = mdp.rewards(i, j)
            mdp.ep.belief.energy[j] = 0.05
            mdp.ep.belief.queue_frac[j] = 0.95
            self.assertEqual(mdp.encode(i), key)
            self.assertFalse(np.allclose(before, mdp.rewards(i, j)))

    def test_reward_prefers_available_buffer(self):
        mdp = self.mdp()
        i = self.sources[0]
        j = next(a for a in mdp.actions(i) if a >= 0)
        mdp.ep.belief.queue_frac[j] = 0.1
        less_congested = mdp.rewards(i, j)[1]
        mdp.ep.belief.queue_frac[j] = 0.9
        self.assertGreater(less_congested, mdp.rewards(i, j)[1])

    def test_observation_delay_changes_report(self):
        observer = ControllerObserver(self.topo, delay_epochs=1)
        n = self.topo.n_nodes
        zero, prr, alive = np.zeros(n), np.ones((n, n)), np.ones(n, dtype=bool)
        observer.observe(np.ones(n), zero, zero, prr, alive)
        report = observer.observe(np.full(n, 0.5), zero, zero, prr, alive)
        np.testing.assert_allclose(report.energy, 1.0)
        self.assertEqual(report.age_epochs, 1)

    def test_link_failure_removes_relay_action(self):
        mdp = self.mdp()
        i = self.sources[0]
        j = next(a for a in mdp.actions(i) if a >= 0)
        self.sim.link_up[i, j] = False
        self.assertNotIn(j, mdp.actions(i))


if __name__ == "__main__":
    unittest.main()

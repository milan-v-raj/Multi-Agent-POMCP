"""Regression tests for the flow simulator's ARQ expected-attempt model."""
import unittest
import numpy as np

from dos_rl import NetworkSimulator, SimConfig, deploy_topology
from dos_rl.policy import GreedyHopPolicy


class PaperFidelityTests(unittest.TestCase):
    def make_scene(self, max_retx=3):
        cfg = SimConfig().with_(max_retx=max_retx)
        topo = deploy_topology(cfg.deployment, np.random.default_rng(3000))
        sim = NetworkSimulator(topo, cfg)
        mask = sim.eligible()
        next_hop, hops = GreedyHopPolicy(cfg).next_hops(sim, [], np.array([]), mask)
        i = next(k for k in range(sim.n) if next_hop[k] >= 0 and hops[k] > 0)
        return sim, mask, next_hop, hops, i, int(next_hop[i])

    def test_expected_attempts_match_truncated_geometric_distribution(self):
        for limit in (1, 2, 3, 5):
            sim, mask, next_hop, hops, i, j = self.make_scene(limit)
            for p in (1e-6, 0.1, 0.5, 0.9, 0.999999, 0.9999999, 1.0):
                with self.subTest(limit=limit, p=p):
                    sim.prr[i, j] = p
                    outcome = sim.evaluate(next_hop, hops, [i], np.array([10.0]), mask)
                    expected = sum((1.0 - p) ** k for k in range(limit))
                    self.assertAlmostEqual(outcome.attempts[i], expected, places=10)
                    self.assertAlmostEqual(outcome.tx_rate[i], outcome.lambda_out[i] * expected)
                    self.assertAlmostEqual(sim._hop_success(i, j), 1 - (1 - p) ** limit)

    def test_gateway_hop_uses_same_attempt_formula(self):
        sim, mask, next_hop, hops, _, _ = self.make_scene()
        i = next(k for k in range(sim.n) if mask[k] and next_hop[k] < 0 and hops[k] == 1)
        sim.radio.prr = lambda distance: 0.9
        outcome = sim.evaluate(next_hop, hops, [i], np.array([10.0]), mask)
        self.assertAlmostEqual(outcome.attempts[i], 1.11)
        self.assertAlmostEqual(outcome.tx_rate[i], outcome.lambda_out[i] * 1.11)

    def test_attempts_decrease_with_increasing_reliability(self):
        sim, mask, next_hop, hops, i, j = self.make_scene()
        actual = []
        for p in (0.1, 0.5, 0.9):
            sim.prr[i, j] = p
            actual.append(sim.evaluate(next_hop, hops, [i], np.array([10.0]), mask).attempts[i])
        np.testing.assert_allclose(actual, [2.71, 1.75, 1.11])
        self.assertTrue(actual[0] > actual[1] > actual[2])


if __name__ == '__main__':
    unittest.main()

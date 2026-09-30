import os
import sys
import time
import numpy as np

# Ensure parent directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from deep_pomcp_env import make_env, ScenarioConfig, PursuitEvasionEnv, StatsLogger

def test_gym_api_compliance():
    env = make_env(density_preset="dense_maze", num_hunters=2)
    obs, info = env.reset(seed=42)

    assert "agent_0" in obs and "agent_1" in obs
    assert obs["agent_0"].shape == (136,)
    assert obs["agent_1"].shape == (136,)
    assert not np.isnan(obs["agent_0"]).any()

    # Step environment
    actions = {"agent_0": 0, "agent_1": 2}
    next_obs, rewards, terminated, truncated, info = env.step(actions)

    assert "agent_0" in next_obs and "agent_1" in next_obs
    assert "agent_0" in rewards and "agent_1" in rewards and "__all__" in rewards
    assert isinstance(terminated["__all__"], bool)
    assert isinstance(truncated["__all__"], bool)
    assert "evader_ground_truth" in info
    assert "belief_mean" in info
    print("[PASS] Gymnasium API compliance test passed.")

def test_3_hunter_configuration():
    env = make_env(density_preset="moderate", num_hunters=3)
    obs, info = env.reset(seed=100)

    assert len(obs) == 3
    assert obs["agent_0"].shape == (140,)
    assert obs["agent_1"].shape == (140,)
    assert obs["agent_2"].shape == (140,)

    actions = {"agent_0": 1, "agent_1": 3, "agent_2": 4}
    next_obs, rewards, term, trunc, info = env.step(actions)
    assert len(rewards) == 4
    print("[PASS] 3-Hunter scaling test passed.")

def test_deterministic_reproducibility():
    env1 = make_env(density_preset="dense_maze", seed=777)
    env2 = make_env(density_preset="dense_maze", seed=777)

    obs1, _ = env1.reset(seed=777)
    obs2, _ = env2.reset(seed=777)

    np.testing.assert_allclose(obs1["agent_0"], obs2["agent_0"])
    np.testing.assert_allclose(obs1["agent_1"], obs2["agent_1"])

    for _ in range(50):
        action = {"agent_0": 3, "agent_1": 1}
        o1, r1, _, _, _ = env1.step(action)
        o2, r2, _, _, _ = env2.step(action)
        np.testing.assert_allclose(o1["agent_0"], o2["agent_0"])
        np.testing.assert_allclose(r1["agent_0"], r2["agent_0"])

    print("[PASS] Deterministic seed reproducibility test passed.")

def test_particle_array_for_pointnet():
    env = make_env(density_preset="dense_maze")
    env.reset(seed=123)
    p_arr = env.get_particle_array()

    assert p_arr.shape == (200, 4)
    assert not np.isnan(p_arr).any()
    print("[PASS] PointNet particle array export test passed.")

def test_stats_logger():
    logger = StatsLogger()
    logger.start_episode(1)
    for _ in range(20):
        logger.log_step(
            hunter_positions=[np.array([100.0, 100.0]), np.array([200.0, 200.0])],
            evader_pos=np.array([400.0, 400.0]),
            belief_mean=np.array([395.0, 402.0]),
            planning_latency_ms=12.5,
            wall_hit=False
        )
    stat = logger.end_episode(success=True, outcome="CAPTURED")
    assert stat.steps == 20
    assert stat.success is True
    summary = logger.compute_summary()
    assert summary["success_rate_pct"] == 100.0
    print("[PASS] StatsLogger integration test passed.")

def test_headless_simulation_speed():
    env = make_env(density_preset="dense_maze", num_hunters=2)
    env.reset(seed=42)

    total_steps = 3000
    start_time = time.perf_counter()
    for _ in range(total_steps):
        actions = {"agent_0": 3, "agent_1": 0}
        _, _, term, trunc, _ = env.step(actions)
        if term["__all__"] or trunc["__all__"]:
            env.reset()

    elapsed = time.perf_counter() - start_time
    sps = total_steps / elapsed
    print(f"[PASS] Headless Throughput: {sps:.1f} steps/second ({total_steps} steps in {elapsed:.2f}s).")
    assert sps > 500.0, f"Expected >500 SPS, got {sps:.1f}"

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING DEEP-POMCP ENVIRONMENT TEST SUITE")
    print("=" * 60)
    test_gym_api_compliance()
    test_3_hunter_configuration()
    test_deterministic_reproducibility()
    test_particle_array_for_pointnet()
    test_stats_logger()
    test_headless_simulation_speed()
    print("=" * 60)
    print("ALL ENVIRONMENT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


"""
Orbital Pursuit-Defense Benchmark Suite (Two-Phase Full-Horizon Evaluation)
Runs full-horizon game episodes between POMCP Pursuer and Intelligent Defender,
measuring cumulative duration in target safe zone (t_s^P) vs interception zone (t_s^D)
matching Tables 6 & 7 of the research paper.
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from orbital_env import OrbitalPursuitDefenseEnv
from particle_filter_3d import ParticleFilter3D
from orbital_pomcp import run_orbital_pomcp
from defender_policy import IntelligentDefenderPolicy


def run_single_episode(seed=None, t_max_hours=24.0, verbose=False):
    """Runs a full-horizon orbital pursuit-defense episode."""
    env = OrbitalPursuitDefenseEnv(t_max_hours=t_max_hours, enable_perturbations=True)
    obs = env.reset(random_seed=seed)
    
    # Initialize Pursuer's 6-DoF Particle Filter tracking Defender
    pf_pursuer = ParticleFilter3D(num_particles=200)
    pf_pursuer.initialize_around_state(
        mean_r=obs["pursuer"]["obs_defender_r"],
        mean_v=np.zeros(3),
        r_spread=1000.0,
        v_spread=0.5
    )
    
    defender_agent = IntelligentDefenderPolicy()
    
    next_p_decision = 0.0
    next_d_decision = 0.0
    
    mcts_latencies = []
    
    if verbose:
        print(f"\n=======================================================")
        print(f"   STARTING 3D ORBITAL ENCOUNTER (Seed: {seed})")
        print(f"   Mission Horizon: {t_max_hours:.1f} hours ({t_max_hours*3600.0:.0f} s)")
        print(f"=======================================================")
        print(f"Initial Pursuer Pos (km): {env.pursuer.r / 1000.0} | Dist to Target: {np.linalg.norm(env.pursuer.r)/1000.0:.2f} km")
        print(f"Initial Defender Pos (km): {env.defender.r / 1000.0} | Dist to Pursuer: {np.linalg.norm(env.pursuer.r - env.defender.r)/1000.0:.2f} km\n")

    while True:
        done, status, d_pt, d_pd, t_s_p, t_s_d = env.check_game_status()
        if done:
            break
            
        current_time = env.current_time
        
        # 1. Pursuer Decision (Two-Phase POMCP)
        if current_time >= next_p_decision:
            obs_dict = env._get_observation()
            pf_pursuer.update(obs_dict["pursuer"]["obs_defender_r"], obs_noise_std=50.0)
            
            t0 = time.perf_counter()
            dt_p, dv_p, act_name = run_orbital_pomcp(
                env.pursuer.r, env.pursuer.v, env.pursuer.fuel_remaining, pf_pursuer, n_sims=150
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            mcts_latencies.append(lat_ms)
            
            actual_dv = env.pursuer.apply_impulse(dv_p, current_time)
            next_p_decision = current_time + dt_p
            
            if verbose:
                phase_label = "DOGFIGHT" if d_pt <= env.D_T else "APPROACH"
                print(f"[{current_time/3600.0:5.2f}h | {phase_label:8s}] PURSUER: {act_name:24s} | dV: {np.linalg.norm(actual_dv):.2f} m/s | Plan: {lat_ms:5.1f}ms | d_PT: {d_pt/1000.0:6.1f}km | d_PD: {d_pd/1000.0:6.1f}km")

        # 2. Defender Decision (Intelligent Paced Policy)
        if current_time >= next_d_decision:
            obs_dict = env._get_observation()
            dt_d, dv_d = defender_agent.get_action(obs_dict["defender"], env.defender.fuel_remaining, current_time)
            actual_dv_d = env.defender.apply_impulse(dv_d, current_time)
            next_d_decision = current_time + dt_d
            
            if verbose:
                print(f"[{current_time/3600.0:5.2f}h | {defender_agent.phase:8s}] DEFENDER: Paced Burn ({dt_d/3600.0:.1f}h) | dV: {np.linalg.norm(actual_dv_d):.2f} m/s | Fuel Rem: {env.defender.fuel_remaining:4.1f} m/s")

        # 3. Determine next event step
        next_event_time = min(next_p_decision, next_d_decision, current_time + env.pursuer.t_sam, env.t_max)
        dt_step = max(1.0, next_event_time - current_time)
        
        # 4. Continuous Physics Propagation
        pf_pursuer.predict(dt_step)
        env.step_ballistic(dt_step)

    # End of episode statistics
    final_done, final_status, final_d_pt, final_d_pd, final_ts_p, final_ts_d = env.check_game_status()
    fuel_used_p = env.pursuer.total_fuel - env.pursuer.fuel_remaining
    fuel_used_d = env.defender.total_fuel - env.defender.fuel_remaining
    avg_latency = np.mean(mcts_latencies) if mcts_latencies else 0.0
    
    result = {
        "status": final_status,
        "time_hours": env.current_time / 3600.0,
        "t_s_p_hours": final_ts_p / 3600.0,
        "t_s_d_hours": final_ts_d / 3600.0,
        "final_d_pt_km": final_d_pt / 1000.0,
        "final_d_pd_km": final_d_pd / 1000.0,
        "fuel_used_p": fuel_used_p,
        "fuel_used_d": fuel_used_d,
        "num_burns_p": len(env.pursuer.impulse_history),
        "num_burns_d": len(env.defender.impulse_history),
        "avg_mcts_latency_ms": avg_latency,
        "trajectory_p": env.pursuer.trajectory_history,
        "trajectory_d": env.defender.trajectory_history
    }
    
    if verbose:
        print(f"\n=======================================================")
        print(f"               MISSION OUTCOME SUMMARY                 ")
        print(f"=======================================================")
        print(f"Final Status            : {final_status}")
        print(f"Pursuer Success Time t_s^P: {result['t_s_p_hours']:.2f} hours (in Target safe zone)")
        print(f"Defender Intercept t_s^D: {result['t_s_d_hours']:.2f} hours (in Intercept bubble)")
        print(f"Pursuer Fuel Consumed   : {fuel_used_p:.2f} / 20.0 m/s ({result['num_burns_p']} burns)")
        print(f"Defender Fuel Consumed  : {fuel_used_d:.2f} / 15.0 m/s ({result['num_burns_d']} burns)")
        print(f"Average Planning Latency: {avg_latency:.2f} ms")
        print(f"=======================================================\n")
        
    return result


def plot_benchmark_results(results, save_path="benchmark_24h_metrics.png"):
    """Generates benchmark visualization matching paper metrics."""
    episodes = np.arange(1, len(results) + 1)
    ts_p = [r["t_s_p_hours"] for r in results]
    ts_d = [r["t_s_d_hours"] for r in results]
    fuel_p = [r["fuel_used_p"] for r in results]
    fuel_d = [r["fuel_used_d"] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0f172a')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', alpha=0.3, color='#475569')
        for spine in ax.spines.values():
            spine.set_color('#475569')
            
    # Subplot 1: Cumulative Success Duration (t_s^P vs t_s^D)
    width = 0.35
    ax1.bar(episodes - width/2, ts_p, width, label='Pursuer in Target Safe Zone (t_s^P)', color='#38bdf8', alpha=0.9)
    ax1.bar(episodes + width/2, ts_d, width, label='Defender in Intercept Bubble (t_s^D)', color='#f43f5e', alpha=0.9)
    ax1.set_title("Cumulative Duration per Episode (Paper Tables 6 & 7)", color='white', fontsize=12, fontweight='bold')
    ax1.set_xlabel("Episode Seed Index", color='white')
    ax1.set_ylabel("Duration (Hours)", color='white')
    ax1.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white')
    
    # Subplot 2: Fuel Consumption
    ax2.plot(episodes, fuel_p, 'o-', color='#38bdf8', linewidth=2, label='Pursuer Delta-V (Max 20.0 m/s)')
    ax2.plot(episodes, fuel_d, 's-', color='#f43f5e', linewidth=2, label='Defender Delta-V (Max 15.0 m/s)')
    ax2.axhline(20.0, color='#38bdf8', linestyle=':', alpha=0.6)
    ax2.axhline(15.0, color='#f43f5e', linestyle=':', alpha=0.6)
    ax2.set_title("24-Hour Fuel Budget Utilization", color='white', fontsize=12, fontweight='bold')
    ax2.set_xlabel("Episode Seed Index", color='white')
    ax2.set_ylabel("Fuel Consumed (m/s)", color='white')
    ax2.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white')
    
    # Subplot 3: Success Time Distribution
    ax3.hist(ts_p, bins=10, color='#38bdf8', alpha=0.8, edgecolor='white', label='t_s^P Distribution')
    ax3.axvline(np.mean(ts_p), color='#fbbf24', linestyle='--', linewidth=2, label=f'Mean t_s^P = {np.mean(ts_p):.2f}h')
    ax3.set_title("Distribution of Pursuer Safe-Zone Loitering Time", color='white', fontsize=12, fontweight='bold')
    ax3.set_xlabel("Hours in Target Zone", color='white')
    ax3.set_ylabel("Frequency", color='white')
    ax3.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white')
    
    # Subplot 4: Outcome Summary Pie Chart
    statuses = [r["status"] for r in results]
    unique_statuses, counts = np.unique(statuses, return_counts=True)
    colors_dict = {
        'PURSUER_DOMINANT_WIN': '#38bdf8',
        'DEFENDER_DOMINANT_WIN': '#f43f5e',
        'DRAW_TIMEOUT': '#94a3b8'
    }
    chart_colors = [colors_dict.get(s, '#3b82f6') for s in unique_statuses]
    
    wedges, texts, autotexts = ax4.pie(
        counts, labels=unique_statuses, autopct='%1.1f%%',
        colors=chart_colors, startangle=140,
        textprops=dict(color="white", fontweight="bold")
    )
    ax4.set_title("Mission Outcome Distribution across Benchmark", color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved benchmark figure to '{save_path}'")


def run_benchmark(num_episodes=20, t_max_hours=24.0):
    """Runs a batch evaluation benchmark over multiple randomized episodes."""
    print(f"\n=================================================================")
    print(f"   ORBITAL POMCP BENCHMARK ({num_episodes} EPISODES | {t_max_hours:.0f}h HORIZON)")
    print(f"=================================================================")
    
    results = []
    
    for ep in range(1, num_episodes + 1):
        res = run_single_episode(seed=ep * 42, t_max_hours=t_max_hours, verbose=False)
        results.append(res)
        
        print(f"Episode [{ep:2d}/{num_episodes}] => Status: {res['status']:22s} | t_s^P: {res['t_s_p_hours']:5.2f}h | t_s^D: {res['t_s_d_hours']:5.2f}h | P_Fuel: {res['fuel_used_p']:5.2f}m/s")

    # Aggregate Statistics matching Tables 6 & 7 of the paper
    p_P_success = (sum(1 for r in results if r["t_s_p_hours"] > 0.0) / num_episodes) * 100.0
    t_bar_P = np.mean([r["t_s_p_hours"] for r in results])
    
    p_D_success = (sum(1 for r in results if r["t_s_d_hours"] > 0.0) / num_episodes) * 100.0
    t_bar_D = np.mean([r["t_s_d_hours"] for r in results])
    
    avg_fuel_p = np.mean([r["fuel_used_p"] for r in results])
    avg_fuel_d = np.mean([r["fuel_used_d"] for r in results])
    avg_lat = np.mean([r["avg_mcts_latency_ms"] for r in results])
    
    print("\n=================================================================")
    print("        BENCHMARK SUMMARY (TABLE 6 & 7 METRIC MATCH)            ")
    print("=================================================================")
    print(f"Pursuer Task Success Rate p_P : {p_P_success:.1f}%")
    print(f"Pursuer Avg Success Time t_P  : {t_bar_P:.2f} hours (in Target safe zone)")
    print(f"Defender Intercept Rate p_D   : {p_D_success:.1f}%")
    print(f"Defender Avg Intercept Time t_D: {t_bar_D:.2f} hours (in Intercept bubble)")
    print(f"Avg Pursuer Fuel Used         : {avg_fuel_p:.2f} / 20.0 m/s")
    print(f"Avg Defender Fuel Used        : {avg_fuel_d:.2f} / 15.0 m/s")
    print(f"Avg MCTS Planning Latency     : {avg_lat:.2f} ms")
    print("=================================================================\n")
    
    plot_benchmark_results(results, save_path="benchmark_24h_metrics.png")
    return results


if __name__ == "__main__":
    run_benchmark(num_episodes=20, t_max_hours=24.0)

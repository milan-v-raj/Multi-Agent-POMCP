"""
3D Orbital Pursuit-Defense Dual-Phase Visualizer
Simulates a full 24-hour encounter and generates publication-grade 3D trajectory
and continuous 24-hour distance metric plots matching Figures 11 and 12 of the research paper.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from orbital_env import OrbitalPursuitDefenseEnv
from particle_filter_3d import ParticleFilter3D
from orbital_pomcp import run_orbital_pomcp
from defender_policy import IntelligentDefenderPolicy


def simulate_and_visualize(seed=42, t_max_hours=24.0):
    env = OrbitalPursuitDefenseEnv(t_max_hours=t_max_hours, enable_perturbations=True)
    obs = env.reset(random_seed=seed)
    
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
    
    time_series = []
    d_pt_series = []
    d_pd_series = []
    d_dt_series = []
    
    print(f"=======================================================")
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
        d_dt = np.linalg.norm(env.defender.r)
        
        time_series.append(current_time / 3600.0)
        d_pt_series.append(d_pt / 1000.0)
        d_pd_series.append(d_pd / 1000.0)
        d_dt_series.append(d_dt / 1000.0)
        
        # Pursuer Decision
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
            
            phase_label = "DOGFIGHT" if d_pt <= env.D_T else "APPROACH"
            print(f"[{current_time/3600.0:5.2f}h | {phase_label:8s}] PURSUER: {act_name:24s} | dV: {np.linalg.norm(actual_dv):.2f} m/s | Plan: {lat_ms:5.1f}ms | d_PT: {d_pt/1000.0:6.1f}km | d_PD: {d_pd/1000.0:6.1f}km")

        # Defender Decision
        if current_time >= next_d_decision:
            obs_dict = env._get_observation()
            dt_d, dv_d = defender_agent.get_action(obs_dict["defender"], env.defender.fuel_remaining, current_time)
            actual_dv_d = env.defender.apply_impulse(dv_d, current_time)
            next_d_decision = current_time + dt_d
            
            print(f"[{current_time/3600.0:5.2f}h | {defender_agent.phase:8s}] DEFENDER: Paced Burn ({dt_d/3600.0:.1f}h) | dV: {np.linalg.norm(actual_dv_d):.2f} m/s | Fuel Rem: {env.defender.fuel_remaining:4.1f} m/s")

        next_event_time = min(next_p_decision, next_d_decision, current_time + env.pursuer.t_sam, env.t_max)
        dt_step = max(1.0, next_event_time - current_time)
        
        pf_pursuer.predict(dt_step)
        env.step_ballistic(dt_step)

    final_done, final_status, final_d_pt, final_d_pd, final_ts_p, final_ts_d = env.check_game_status()
    fuel_used_p = env.pursuer.total_fuel - env.pursuer.fuel_remaining
    fuel_used_d = env.defender.total_fuel - env.defender.fuel_remaining
    avg_latency = np.mean(mcts_latencies) if mcts_latencies else 0.0
    
    print(f"\n=======================================================")
    print(f"               MISSION OUTCOME SUMMARY                 ")
    print(f"=======================================================")
    print(f"Final Status            : {final_status}")
    print(f"Pursuer Success Time t_s^P: {final_ts_p/3600.0:.2f} hours (in Target safe zone)")
    print(f"Defender Intercept t_s^D: {final_ts_d/3600.0:.2f} hours (in Intercept bubble)")
    print(f"Pursuer Fuel Consumed   : {fuel_used_p:.2f} / 20.0 m/s ({len(env.pursuer.impulse_history)} burns)")
    print(f"Defender Fuel Consumed  : {fuel_used_d:.2f} / 15.0 m/s ({len(env.defender.impulse_history)} burns)")
    print(f"Average Planning Latency: {avg_latency:.2f} ms")
    print(f"=======================================================\n")

    # --- PLOT 1: 3D Trajectory in LVLH Frame (Fig 11 match) ---
    traj_p = np.array([pt[1] / 1000.0 for pt in env.pursuer.trajectory_history])
    traj_d = np.array([pt[1] / 1000.0 for pt in env.defender.trajectory_history])
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Plot orbits
    ax.plot(traj_p[:, 0], traj_p[:, 1], traj_p[:, 2], color='#38bdf8', linewidth=2.5, label='Pursuer Trajectory (POMCP)')
    ax.plot(traj_d[:, 0], traj_d[:, 1], traj_d[:, 2], color='#f43f5e', linewidth=2.0, linestyle='--', label='Defender Trajectory')
    
    # Plot Target at origin
    ax.scatter([0], [0], [0], color='#fbbf24', s=180, marker='*', label='Target Spacecraft (Origin)')
    
    # Initial / Final markers
    ax.scatter([traj_p[0, 0]], [traj_p[0, 1]], [traj_p[0, 2]], color='#38bdf8', marker='o', s=80, label='Pursuer Start')
    ax.scatter([traj_p[-1, 0]], [traj_p[-1, 1]], [traj_p[-1, 2]], color='#0284c7', marker='^', s=100, label='Pursuer End (24h)')
    
    ax.scatter([traj_d[0, 0]], [traj_d[0, 1]], [traj_d[0, 2]], color='#f43f5e', marker='o', s=80, label='Defender Start')
    ax.scatter([traj_d[-1, 0]], [traj_d[-1, 1]], [traj_d[-1, 2]], color='#e11d48', marker='^', s=100, label='Defender End (24h)')
    
    # 20km Safe Zone Sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = 20.0 * np.cos(u) * np.sin(v)
    ys = 20.0 * np.sin(u) * np.sin(v)
    zs = 20.0 * np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color='#fbbf24', alpha=0.15, label='Target Safe Zone (20 km)')
    
    ax.set_xlabel('Radial x (km)', color='white', labelpad=10)
    ax.set_ylabel('Along-Track y (km)', color='white', labelpad=10)
    ax.set_zlabel('Cross-Track z (km)', color='white', labelpad=10)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.25, color='#475569')
    ax.set_title("3D Relative Orbital Motion (LVLH Frame) Over 24 Hours", color='white', fontsize=14, fontweight='bold', pad=15)
    ax.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='white', loc='upper right')
    
    traj_path = "orbital_demo_3d_two_phases.png"
    plt.savefig(traj_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved two-phase 3D trajectory figure to '{traj_path}'")

    # --- PLOT 2: Continuous 24h Distance Metrics (Fig 12 match) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    ax.plot(time_series, d_pt_series, color='#38bdf8', linewidth=2.5, label=r'Pursuer-Target Distance $d_{PT}$')
    ax.plot(time_series, d_pd_series, color='#f43f5e', linewidth=2.0, linestyle='--', label=r'Pursuer-Defender Distance $d_{PD}$')
    ax.plot(time_series, d_dt_series, color='#fbbf24', linewidth=1.5, linestyle=':', label=r'Defender-Target Distance $d_{DT}$')
    
    ax.axhline(20.0, color='#38bdf8', linestyle=':', linewidth=1.5, alpha=0.8, label=r'Target Safe Boundary $D_T = 20\,\mathrm{km}$')
    ax.axhline(10.0, color='#f43f5e', linestyle=':', linewidth=1.5, alpha=0.8, label=r'Interception Bubble $D_P = 10\,\mathrm{km}$')
    
    # Highlight Pursuer Success Holding region
    ax.fill_between(time_series, 0, d_pt_series, where=(np.array(d_pt_series) <= 20.0), color='#38bdf8', alpha=0.25, label=r'Pursuer Safe Zone Holding ($t_s^P$)')
    
    ax.set_xlabel('Mission Time (Hours)', color='white', fontsize=12)
    ax.set_ylabel('Distance (km)', color='white', fontsize=12)
    ax.set_title(f"24-Hour Continuous Distance Curves & Tactical Hold (t_s^P = {final_ts_p/3600.0:.2f}h, t_s^D = {final_ts_d/3600.0:.2f}h)", color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='#475569')
    for spine in ax.spines.values():
        spine.set_color('#475569')
        
    ax.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white', loc='upper right')
    
    dist_path = "orbital_demo_distance_metrics_24h.png"
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 24h distance metrics figure to '{dist_path}'")


if __name__ == "__main__":
    simulate_and_visualize(seed=42, t_max_hours=24.0)

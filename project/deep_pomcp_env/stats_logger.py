import csv
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class EpisodeStats:
    episode_idx: int
    success: bool
    outcome: str
    steps: int
    min_inter_agent_dist: float
    mean_rmse: float
    max_rmse: float
    wall_collisions: int
    avg_planning_latency_ms: float
    total_time_ms: float

class StatsLogger:
    """Tracks per-step telemetry and summarizes episode benchmarks."""

    def __init__(self, output_csv: Optional[str] = None):
        self.output_csv = output_csv
        self.episodes: List[EpisodeStats] = []
        
        # Current active episode buffers
        self._current_ep_idx = 0
        self._step_count = 0
        self._rmse_history: List[float] = []
        self._inter_agent_dists: List[float] = []
        self._planning_latencies: List[float] = []
        self._wall_collisions = 0
        self._ep_start_time = 0.0

    def start_episode(self, episode_idx: int):
        self._current_ep_idx = episode_idx
        self._step_count = 0
        self._rmse_history.clear()
        self._inter_agent_dists.clear()
        self._planning_latencies.clear()
        self._wall_collisions = 0
        self._ep_start_time = time.perf_counter()

    def log_step(self, hunter_positions: List[np.ndarray], evader_pos: np.ndarray,
                 belief_mean: Optional[np.ndarray] = None,
                 planning_latency_ms: Optional[float] = None,
                 wall_hit: bool = False):
        self._step_count += 1

        # Track Particle Filter RMSE
        if belief_mean is not None:
            rmse = float(np.linalg.norm(belief_mean - evader_pos))
            self._rmse_history.append(rmse)

        # Track Inter-Agent Distance
        if len(hunter_positions) >= 2:
            min_dist = float('inf')
            for i in range(len(hunter_positions)):
                for j in range(i + 1, len(hunter_positions)):
                    d = float(np.linalg.norm(hunter_positions[i] - hunter_positions[j]))
                    if d < min_dist:
                        min_dist = d
            self._inter_agent_dists.append(min_dist)

        # Track Planning Latency
        if planning_latency_ms is not None:
            self._planning_latencies.append(planning_latency_ms)

        if wall_hit:
            self._wall_collisions += 1

    def end_episode(self, success: bool, outcome: str = "CAPTURED") -> EpisodeStats:
        ep_duration_ms = (time.perf_counter() - self._ep_start_time) * 1000.0

        min_inter_dist = min(self._inter_agent_dists) if self._inter_agent_dists else 0.0
        mean_rmse = float(np.mean(self._rmse_history)) if self._rmse_history else 0.0
        max_rmse = float(np.max(self._rmse_history)) if self._rmse_history else 0.0
        avg_latency = float(np.mean(self._planning_latencies)) if self._planning_latencies else 0.0

        stat = EpisodeStats(
            episode_idx=self._current_ep_idx,
            success=success,
            outcome=outcome,
            steps=self._step_count,
            min_inter_agent_dist=min_inter_dist,
            mean_rmse=mean_rmse,
            max_rmse=max_rmse,
            wall_collisions=self._wall_collisions,
            avg_planning_latency_ms=avg_latency,
            total_time_ms=ep_duration_ms
        )
        self.episodes.append(stat)

        if self.output_csv:
            self._append_to_csv(stat)

        return stat

    def compute_summary(self) -> Dict[str, Any]:
        if not self.episodes:
            return {}

        n = len(self.episodes)
        successes = [e for e in self.episodes if e.success]
        success_rate = (len(successes) / n) * 100.0

        ttc_list = [e.steps for e in successes]
        mean_ttc = float(np.mean(ttc_list)) if ttc_list else float('nan')
        std_ttc = float(np.std(ttc_list)) if ttc_list else float('nan')

        all_latencies = [e.avg_planning_latency_ms for e in self.episodes if e.avg_planning_latency_ms > 0]
        mean_latency = float(np.mean(all_latencies)) if all_latencies else 0.0
        std_latency = float(np.std(all_latencies)) if all_latencies else 0.0

        all_rmse = [e.mean_rmse for e in self.episodes if e.mean_rmse > 0]
        mean_rmse = float(np.mean(all_rmse)) if all_rmse else 0.0

        return {
            "total_episodes": n,
            "success_rate_pct": success_rate,
            "mean_time_to_capture_steps": mean_ttc,
            "std_time_to_capture_steps": std_ttc,
            "mean_planning_latency_ms": mean_latency,
            "std_planning_latency_ms": std_latency,
            "mean_belief_rmse_px": mean_rmse
        }

    def print_summary(self):
        s = self.compute_summary()
        if not s:
            print("[StatsLogger] No episode data recorded.")
            return

        print("\n" + "=" * 50)
        print("=== MULTI-AGENT PURSUIT BENCHMARK SUMMARY ===")
        print("=" * 50)
        print(f"Total Episodes Run    : {s['total_episodes']}")
        print(f"Capture Success Rate  : {s['success_rate_pct']:.1f}%")
        print(f"Mean Time-to-Capture  : {s['mean_time_to_capture_steps']:.1f} ± {s['std_time_to_capture_steps']:.1f} steps")
        print(f"Mean Planning Latency : {s['mean_planning_latency_ms']:.2f} ± {s['std_planning_latency_ms']:.2f} ms")
        print(f"Mean Belief RMSE      : {s['mean_belief_rmse_px']:.2f} px")
        print("=" * 50 + "\n")

    def _append_to_csv(self, stat: EpisodeStats):
        file_exists = False
        try:
            with open(self.output_csv, 'r') as f:
                file_exists = True
        except FileNotFoundError:
            file_exists = False

        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "episode", "success", "outcome", "steps",
                    "min_inter_dist", "mean_rmse", "max_rmse",
                    "wall_collisions", "avg_latency_ms", "duration_ms"
                ])
            writer.writerow([
                stat.episode_idx, stat.success, stat.outcome, stat.steps,
                f"{stat.min_inter_agent_dist:.2f}", f"{stat.mean_rmse:.2f}",
                f"{stat.max_rmse:.2f}", stat.wall_collisions,
                f"{stat.avg_planning_latency_ms:.2f}", f"{stat.total_time_ms:.2f}"
            ])


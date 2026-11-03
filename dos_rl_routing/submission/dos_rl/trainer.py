"""Training and evaluation loops for route-computation episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agents.linear_fa import LinearQAgent
from .agents.tabular import MonteCarloAgent, QLearningAgent, SarsaAgent, TabularAgent
from .config import RLConfig
from .mdp import (
    TERM_DEAD_END,
    TERM_EXHAUSTED,
    TERM_GOAL,
    TERM_HOP_LIMIT,
    TERM_LOOP,
    ActionResult,
    RoutingMDP,
)


@dataclass
class EpisodeStats:
    episode: int
    return_: float
    objective_return: List[float]
    length: int
    terminal: str
    hops: int
    epsilon: float
    success: int

    def as_row(self) -> dict:
        row = {
            "episode": self.episode,
            "return": self.return_,
            "length": self.length,
            "terminal": self.terminal,
            "success": self.success,
            "epsilon": self.epsilon,
        }
        for k, v in enumerate(self.objective_return):
            row[f"return_o{k + 1}"] = v
        return row


def _scalar(rl: RLConfig, rvec: np.ndarray) -> float:
    w = np.asarray(rl.reward_weights, dtype=float)
    return float(np.dot(w, rvec) / max(w.sum(), 1e-12))


# ---------------------------------------------------------------------------
def run_episode(
    mdp: RoutingMDP,
    agent,
    rl: RLConfig,
    source: int,
    rng: np.random.Generator,
    train: bool = True,
    energy_spread: float = 0.35,
) -> EpisodeStats:
    """Roll out one route-computation episode."""
    mdp.sample_episode(source, rng, energy_spread=energy_spread)
    state = mdp.reset(source)
    node = mdp.current

    total = 0.0
    obj_total = np.zeros(3)
    length = 0
    terminal = TERM_HOP_LIMIT
    trajectory: List[Tuple[tuple, int, np.ndarray, float]] = []

    is_mc = isinstance(agent, MonteCarloAgent)
    is_sarsa = isinstance(agent, SarsaAgent)
    is_linear = isinstance(agent, LinearQAgent)

    # SARSA is on-policy: it needs the action it is about to execute
    pending_action: Optional[int] = None
    if is_sarsa:
        acts0 = mdp.actions(node)
        if acts0:
            ia = agent.epsilon_greedy(state, acts0, rng) if train else agent.greedy(state, acts0)
            pending_action = int(acts0[int(ia)])

    while True:
        acts = mdp.actions(node)
        if not acts:
            terminal = TERM_DEAD_END
            break

        if is_linear:
            feats = np.stack([mdp.ep.belief.node_features(node, a, gateway_prr=mdp._bs_prr(node)) for a in acts])
            idx = agent.act(feats, state, acts, rng, greedy=not train)
        elif is_sarsa:
            if pending_action is not None and pending_action in acts:
                action = pending_action
                pending_action = None
                res = mdp.step(action, rng)
                total += res.scalar_reward
                obj_total += res.total_for_objective
                length += 1
                if res.done:
                    terminal = res.terminal
                    if train:
                        agent.update_transition(
                            state, action, res.total_for_objective, None, None, True,
                            float(res.info.get("terminal_reward", 0.0)),
                        )
                    break
                nn = res.next_node
                nacts = mdp.actions(nn)
                if not nacts:
                    if train:
                        agent.update_transition(
                            state, action, res.total_for_objective, None, None, True, rl.dead_reward
                        )
                    terminal = TERM_DEAD_END
                    break
                ia = agent.epsilon_greedy(res.next_state, nacts, rng) if train else agent.greedy(res.next_state, nacts)
                na = int(nacts[int(ia)])
                if train:
                    agent.update_transition(state, action, res.total_for_objective, res.next_state, na, False, 0.0)
                state, node = res.next_state, nn
                pending_action = na
                continue
            idx = agent.greedy(state, acts) if not train else agent.epsilon_greedy(state, acts, rng)
        elif is_mc:
            idx = agent.act(state, acts, rng, greedy=not train)
        else:
            idx = agent.epsilon_greedy(state, acts, rng) if train else agent.greedy(state, acts)

        action = acts[idx]
        res: ActionResult = mdp.step(action, rng)
        total += res.scalar_reward
        obj_total += res.total_for_objective
        length += 1

        if is_mc:
            trajectory.append((state, action, res.rewards.copy(), res.shaping))

        if res.done:
            terminal = res.terminal
            if train and is_mc:
                agent.update_episode(trajectory, float(res.info.get("terminal_reward", 0.0)))
            elif train and is_linear:
                agent.update(
                    feats[idx], state, idx, action, res.total_for_objective, None, True,
                    float(res.info.get("terminal_reward", 0.0)),
                )
            elif train and is_sarsa:
                agent.update_transition(
                    state, action, res.total_for_objective, None, None, True,
                    float(res.info.get("terminal_reward", 0.0)),
                )
            elif train:
                agent.update_transition(
                    state, action, res.total_for_objective, None, (), True,
                    float(res.info.get("terminal_reward", 0.0)),
                )
            break

        next_node = res.next_node
        next_acts = mdp.actions(next_node)
        next_state = res.next_state

        if train and not is_mc:
            if is_linear:
                if next_acts:
                    nf = np.stack([mdp.ep.belief.node_features(next_node, b, gateway_prr=mdp._bs_prr(next_node)) for b in next_acts])
                else:
                    nf = None
                agent.update(feats[idx], state, idx, action, res.total_for_objective,
                             nf, not next_acts, rl.dead_reward if not next_acts else 0.0)
            elif is_sarsa:
                if next_acts:
                    na_idx = agent.epsilon_greedy(next_state, next_acts, rng)
                    na = next_acts[na_idx]
                    pending_action = na
                    agent.update_transition(state, action, res.total_for_objective, next_state, na, False, 0.0)
                    state, node = next_state, next_node
                    continue
                else:
                    agent.update_transition(state, action, res.total_for_objective, None, None, True, rl.dead_reward)
                    break
            else:
                agent.update_transition(state, action, res.total_for_objective, next_state, next_acts, False, 0.0)

        state, node = next_state, next_node

    if getattr(agent, "rl", None) is not None and getattr(agent, "decay_epsilon", None):
        if train:
            agent.decay_epsilon()

    return EpisodeStats(
        episode=0,
        return_=float(total),
        objective_return=obj_total.tolist(),
        length=length,
        terminal=terminal,
        hops=length,
        epsilon=float(getattr(agent, "epsilon", 0.0)),
        success=int(terminal == TERM_GOAL),
    )


# ---------------------------------------------------------------------------
def training_origins(mdp: RoutingMDP, sources: Sequence[int], all_nodes: bool = True) -> List[int]:
    """Choose eligible relay origins for training, or use only the given sources."""
    if not all_nodes:
        return list(sources)
    elig = np.where(mdp.ep.eligible)[0] if mdp.ep.eligible.any() else np.arange(mdp.n)
    reach = mdp.topo.dist_bs <= mdp.cfg.deployment.comm_range_m
    cand = [int(i) for i in elig if not reach[i]]
    return cand or list(sources)


def train(
    agent,
    mdp: RoutingMDP,
    sources: Sequence[int],
    episodes: int,
    seed: int = 0,
    eval_every: int = 0,
    eval_episodes: int = 12,
    energy_spread: float = 0.35,
    log: bool = False,
    all_nodes: bool = True,
) -> List[EpisodeStats]:
    """Train ``agent`` for ``episodes`` route computations."""
    rng = np.random.default_rng(seed)
    rl: RLConfig = agent.rl
    history: List[EpisodeStats] = []
    origins = training_origins(mdp, sources, all_nodes=all_nodes)
    for ep in range(episodes):
        src = int(origins[ep % len(origins)])
        st = run_episode(mdp, agent, rl, src, rng, train=True, energy_spread=energy_spread)
        st.episode = ep
        history.append(st)
        if log and (ep + 1) % max(1, episodes // 10) == 0:
            recent = history[-max(1, episodes // 10):]
            print(
                f"  ep {ep + 1:5d}/{episodes} | return {np.mean([s.return_ for s in recent]):+7.3f} "
                f"| success {np.mean([s.success for s in recent]) * 100:5.1f}% "
                f"| eps {agent.epsilon:.3f}",
                flush=True,
            )
    return history


def evaluate(
    agent,
    mdp: RoutingMDP,
    sources: Sequence[int],
    episodes: int = 30,
    seed: int = 999,
    energy_spread: float = 0.35,
) -> Dict[str, float]:
    """Greedy evaluation over ``episodes`` independent route computations."""
    rng = np.random.default_rng(seed)
    rl: RLConfig = agent.rl
    rows = []
    for k in range(episodes):
        src = int(sources[k % len(sources)])
        st = run_episode(mdp, agent, rl, src, rng, train=False, energy_spread=energy_spread)
        rows.append(st)
    succ = np.array([r.success for r in rows], float)
    ret = np.array([r.return_ for r in rows], float)
    ln = np.array([r.length for r in rows], float)
    terms: Dict[str, int] = {}
    for r in rows:
        terms[r.terminal] = terms.get(r.terminal, 0) + 1
    out = {
        "success_rate": float(succ.mean() * 100.0),
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std(ddof=1)) if ret.size > 1 else 0.0,
        "return_ci95": float(1.96 * ret.std(ddof=1) / np.sqrt(ret.size)) if ret.size > 1 else 0.0,
        "length_mean": float(ln.mean()),
        "n_episodes": float(episodes),
    }
    for t in (TERM_GOAL, TERM_HOP_LIMIT, TERM_LOOP, TERM_DEAD_END, TERM_EXHAUSTED):
        out[f"term_{t}"] = float(terms.get(t, 0))
    return out


def summarise(history: List[EpisodeStats], window: int = 20) -> Dict[str, np.ndarray]:
    """Smoothed learning curves for plotting."""
    ret = np.array([h.return_ for h in history], float)
    succ = np.array([h.success for h in history], float)
    ln = np.array([h.length for h in history], float)
    eps = np.array([h.epsilon for h in history], float)
    if ret.size >= window:
        k = np.ones(window) / window
        ret_s = np.convolve(ret, k, mode="valid")
        succ_s = np.convolve(succ, k, mode="valid")
        ln_s = np.convolve(ln, k, mode="valid")
        eps_s = eps[window - 1:]
        x = np.arange(window - 1, ret.size)
    else:
        ret_s, succ_s, ln_s, eps_s, x = ret, succ, ln, eps, np.arange(ret.size)
    return {
        "x": x,
        "return": ret_s,
        "success": succ_s * 100.0,
        "length": ln_s,
        "epsilon": eps_s,
        "return_raw": ret,
    }

"""SDWSN-IoT energy-efficient routing testbed (DOS-RL reproduction + extensions).

Reference framework
-------------------
Godfrey, D.; Suh, B.; Lim, B.H.; Lee, K.-C.; Kim, K.-I.
"An Energy-Efficient Routing Protocol with Reinforcement Learning in
Software-Defined Wireless Sensor Networks", Sensors 2023, 23, 8435.
https://doi.org/10.3390/s23208435

This package is fully self-contained and has no dependency on the Deep-POMCP
capstone code that also lives in this repository.
"""

from .config import SimConfig, RLConfig, DeploymentConfig, default_config
from .topology import Topology, deploy_topology
from .energy import RadioModel
from .network_sim import NetworkSimulator, RouteEval, FlowResult
from .mdp import RoutingMDP, ActionResult
from .observer import ControllerObserver, LinkMetrics
from .belief import BeliefState, BeliefTracker

__all__ = [
    "SimConfig",
    "RLConfig",
    "DeploymentConfig",
    "default_config",
    "Topology",
    "deploy_topology",
    "RadioModel",
    "NetworkSimulator",
    "RouteEval",
    "FlowResult",
    "RoutingMDP",
    "ActionResult",
    "ControllerObserver",
    "LinkMetrics",
    "BeliefState",
    "BeliefTracker",
]

__version__ = "1.0.0"

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
from ..scenarios import Obstacle

class BasePursuerPolicy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int, obstacles: List[Obstacle], width: int, height: int) -> int:
        pass

    def reset(self):
        pass

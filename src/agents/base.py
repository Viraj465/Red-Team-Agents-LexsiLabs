
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    @abstractmethod
    def get_action(self, obs: Dict[str, Any], action_space: Any) -> Any:
        """Selects an action given the current observation and available action space."""
        pass

    @abstractmethod
    def reset(self):
        """Resets internal agent state between episodes."""
        pass
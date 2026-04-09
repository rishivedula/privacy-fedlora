"""Server-side federated learning logic."""

from typing import List


class Server:
    """Federated learning server."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model = None
        self.round = 0

    def initialize_model(self) -> None:
        """Initialize global model."""
        raise NotImplementedError

    def select_clients(self, num_clients: int) -> List[int]:
        """Select clients for current round."""
        raise NotImplementedError

    def aggregate(self, updates: List[dict]) -> None:
        """Aggregate client updates into global model."""
        raise NotImplementedError

    def get_model_state(self) -> dict:
        """Return current global model state."""
        raise NotImplementedError

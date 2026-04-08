"""Client-side federated learning logic."""

from typing import Any


class Client:
    """Federated learning client."""

    def __init__(self, client_id: int, config: dict) -> None:
        self.client_id = client_id
        self.config = config
        self.model = None
        self.data = None

    def set_model(self, model_state: dict) -> None:
        """Receive model state from server."""
        raise NotImplementedError

    def train(self) -> dict:
        """Run local training and return model update."""
        raise NotImplementedError

    def get_update(self) -> dict:
        """Return model update after local training."""
        raise NotImplementedError

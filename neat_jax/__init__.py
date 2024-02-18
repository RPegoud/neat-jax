from .neat_dataclasses import ActivationState, Network
from .nn import (
    forward,
    forward_toggled_nodes,
    get_activation,
    get_active_connections,
    get_required_activations,
    init,
    toggle_receivers,
)
from .utils import plot_network

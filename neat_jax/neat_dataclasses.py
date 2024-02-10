import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ActivationState:
    values: jnp.ndarray
    toggled: jnp.ndarray
    activation_counts: jnp.ndarray


@struct.dataclass
class Network:
    node_indices: jnp.ndarray
    node_types: jnp.ndarray
    edges: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.nodes_indices)

    @property
    def input_nodes(self) -> int:
        return jnp.where(self.node_types == 0)[0]

    @property
    def output_nodes(self) -> int:
        return jnp.where(self.node_types == 2)[0]

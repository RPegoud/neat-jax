import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ActivationState:
    """
    Attributes:
        values (jnp.ndarray): Current node activation values
        toggled (jnp.ndarray): Boolean array indicating which neurons should
        fire at the next step
        activation_counts (jnp.ndarray): Number of times each node received an activation
    """

    values: jnp.ndarray
    toggled: jnp.ndarray
    activation_counts: jnp.ndarray


@struct.dataclass
class Network:
    """
    Stores data relative to a Neat Network's topology.

    Attributes:
        node_indices (jnp.ndarray): Index of each node
        node_types (jnp.ndarray): Type of each node:
            - `0`: Input node
            - `1`: Hidden node
            - `2`: Output node
        edges (jnp.ndarray): Weight of the connection between sender and receiver `i`
        senders (jnp.ndarray): Array of node indices used as inputs to receiver nodes
        receivers (jnp.ndarray): Array of node indices used as outputs to sender nodes
    """

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

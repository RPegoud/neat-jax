from typing import NamedTuple

import jax.numpy as jnp


class Node(NamedTuple):
    index: jnp.int32
    type: jnp.int32
    enabled: jnp.bool_


class ActivationState(NamedTuple):
    activations: jnp.ndarray
    activated_nodes: jnp.ndarray
    activation_counts: jnp.ndarray


class Network(NamedTuple):
    nodes: list[Node]
    edges: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.nodes.index)

    @property
    def input_nodes(self) -> int:
        return jnp.where(self.nodes.type == 1)[0]

    @property
    def n_inputs(self) -> int:
        return len(self.input_nodes)

    @property
    def n_required_activations(self) -> dict:
        """
        The number of required activations for each receiver node.
        """
        unique, counts = jnp.unique(self.receivers, return_counts=True)
        return jnp.zeros(self.n_nodes).at[unique].set(counts)

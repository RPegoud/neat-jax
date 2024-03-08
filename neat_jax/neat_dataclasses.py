from functools import partial

import jax
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
        node_depths (jnp.ndarray): Depth of each node in the network, used to add new connections in
        topological order (and avoid recurrent connections)
        outdated_depths (bool, Optional): Boolean flag indicating whether `node_depths` is up to date
        for the current network topology, usually set to `True` when a mutation adds an edge or a node
    """

    values: jnp.ndarray
    toggled: jnp.ndarray
    activation_counts: jnp.ndarray
    has_fired: jnp.ndarray
    node_depths: jnp.ndarray
    outdated_depths: bool = True

    def __repr__(self) -> str:
        for atr in ActivationState.__dataclass_fields__.keys():
            print(f"{atr}: {self.__getattribute__(atr)}")
        return ""

    @staticmethod
    @partial(jax.jit, static_argnames=("max_nodes"))
    def from_inputs(input_values: jnp.ndarray, max_nodes: int) -> "ActivationState":
        """
        Resets the ActivationState in prevision of a forward pass or a depth scan.

        Args:
            input_values (jnp.ndarray): The activation values of the network's input nodes
            max_nodes (int): The maximum capacity of the network

        Returns:
            ActivationState: The reset ActivationState with:

                - ``values``: initialized based on inputs
                - ``toggled``: input neurons toggled
                - ``activation_counts``: set to zero
                - ``has_fired``: set to zero
                - ``outdated_depths``: True
        """
        return ActivationState(
            values=input_values,
            toggled=jnp.int32(input_values != 0),
            activation_counts=jnp.zeros(max_nodes, dtype=jnp.int32),
            has_fired=jnp.zeros(max_nodes, dtype=jnp.int32),
            node_depths=jnp.zeros(max_nodes, dtype=jnp.int32),
            outdated_depths=True,
        )


@struct.dataclass
class Network:
    """
    Stores data relative to a Neat Network's topology.

    Attributes:
        node_indices (jnp.ndarray): Index of each node
        node_types (jnp.ndarray): Type of each node:
        ```
        - 0: Input node
        - 1: Hidden node
        - 2: Output node
        - 3: Disabled or uninitialized node
        ```

        activation_indices (jnp.ndarray): Mapping index of the activation function used
        by node `i`:

        ```python
        0: 1 / (1 + jnp.exp(-x)),  # sigmoid
        1: jnp.divide(1, x),  # inverse
        2: jnp.sinh(x) / jnp.cosh(x),  # hyperbolic cosine
        3: jnp.float32(jnp.maximum(0, x)),  # relu
        4: jnp.float32(jnp.abs(x)),  # absolute value
        5: jnp.sin(x),  # sine
        6: jnp.exp(jnp.square(-x)),  # gaussian
        7: jnp.float32(jnp.sign(x)),  # step
        ```
        edges (jnp.ndarray): Weight of the connection between sender and receiver `i`
        senders (jnp.ndarray): Array of node indices used as inputs to receiver nodes
        receivers (jnp.ndarray): Array of node indices used as outputs to sender nodes
    """

    node_indices: jnp.ndarray
    node_types: jnp.ndarray
    activation_indices: jnp.ndarray
    weights: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray
    output_size: int

    def __repr__(self) -> str:
        for atr in Network.__dataclass_fields__.keys():
            print(f"{atr}: {self.__getattribute__(atr)}")
        return ""

    @property
    def n_nodes(self) -> int:
        return len(self.node_indices)

    @property
    def n_enabled_connections(self) -> jnp.int32:
        return jnp.int32(sum(self.senders > 0))

    @property
    def n_enabled_nodes(
        self,
    ) -> jnp.int32:
        return jnp.max(jnp.concatenate([self.senders, self.receivers]))

    @property
    def node_type_counts(self) -> jnp.ndarray:
        return jnp.bincount(self.node_types, length=4)

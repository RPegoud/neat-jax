import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Network:
    """
    Stores data relative to a Neat Network's topology.

    Attributes:
        node_indices (chex.Array): Index of each node
        node_types (chex.Array): Type of each node:
        ```
        - 0: Input node
        - 1: Hidden node
        - 2: Output node
        - 3: Disabled or uninitialized node
        ```

        activation_fns (chex.Array): Mapping index of the activation function used
        by node `i`:

        ```python
        0: jnp.float32(x) # identity function
        1: 1 / (1 + jnp.exp(-x)),  # sigmoid
        2: jnp.divide(1, x),  # inverse
        3: jnp.sinh(x) / jnp.cosh(x),  # hyperbolic cosine
        4: jnp.float32(jnp.maximum(0, x)),  # relu
        5: jnp.float32(jnp.abs(x)),  # absolute value
        6: jnp.sin(x),  # sine
        7: jnp.exp(jnp.square(-x)),  # gaussian
        8: jnp.float32(jnp.sign(x)),  # step
        ```
        edges (chex.Array): Weight of the connection between sender and receiver `i`
        senders (chex.Array): Array of node indices used as inputs to receiver nodes
        receivers (chex.Array): Array of node indices used as outputs to sender nodes
    """

    input_size: int
    output_size: int
    max_nodes: int
    node_indices: chex.Array
    node_types: chex.Array
    activation_fns: chex.Array
    senders: chex.Array
    receivers: chex.Array
    weights: chex.Array

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
    def node_type_counts(self) -> chex.Array:
        return jnp.bincount(self.node_types, length=4)


@struct.dataclass
class ActivationState:
    """
    Stores the data related to the network state at runtime.

    Attributes:
        values (chex.Array): Current node activation values
        toggled (chex.Array): Boolean array indicating which neurons should
        fire at the next step
        activation_counts (chex.Array): Number of times each node received an activation
        node_depths (chex.Array): Depth of each node in the network, used to add new connections in
        topological order (and avoid recurrent connections)
        input_mappings (chex.Array): An array of size `max_nodes` where the n-th element points to
        outdated_depths (bool, Optional): Boolean flag indicating whether `node_depths` is up to date
        for the current network topology, usually set to `True` when a mutation adds an edge or a node
    """

    values: chex.Array
    toggled: chex.Array
    activation_counts: chex.Array
    has_fired: chex.Array
    node_depths: chex.Array
    outdated_depths: bool = True

    def __repr__(self) -> str:
        for atr in ActivationState.__dataclass_fields__.keys():
            print(f"{atr}: {self.__getattribute__(atr)}")
        return ""

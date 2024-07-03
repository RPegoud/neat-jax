import chex
import jax
import jax.numpy as jnp

from .neat_dataclasses import ActivationState, Network
from .utils.utils import cartesian_product


def get_initial_activations(inputs: chex.Array, senders: chex.Array) -> chex.Array:
    """Initializes ActivationState.values based on an array of inputs."""
    default_values = jnp.zeros_like(senders)
    keys = jnp.arange(len(inputs))
    matches = jnp.searchsorted(keys, senders)
    mask = (matches < len(keys)) & (keys[matches] == senders)
    return jnp.where(mask, inputs[matches], default_values)


def create_toggled_mask(
    node_types: chex.Array, senders: chex.Array, input_size: int
) -> chex.Array:
    """Initializes the ActivationState.toggled mask based on input nodes"""
    input_nodes_indices = jnp.where(node_types == 0, size=input_size)[0]
    sender_mask = jnp.isin(senders, input_nodes_indices)
    return jnp.int32(sender_mask)


def activation_state_from_inputs(
    inputs: chex.Array,
    senders: chex.Array,
    node_types: chex.Array,
    input_size: int,
    max_nodes: int,
) -> "ActivationState":
    """
    Resets the ActivationState in prevision of a forward pass or a depth scan.

    Args:
        inputs (chex.Array): The activation values of the network's input nodes
        max_nodes (int): The maximum capacity of the network

    Returns:
        ActivationState: The reset ActivationState with:

            - ``values``: initialized based on inputs
            - ``toggled``: input neurons toggled
            - ``activation_counts``: set to zero
            - ``has_fired``: set to zero
            - ``outdated_depths``: True
    """

    values = get_initial_activations(inputs, senders)
    toggled = create_toggled_mask(node_types, senders, input_size)

    return ActivationState(
        values=values,
        toggled=toggled,
        activation_counts=jnp.zeros(max_nodes, dtype=jnp.int32),
        has_fired=jnp.zeros(max_nodes, dtype=jnp.int32),
        node_depths=jnp.zeros(max_nodes, dtype=jnp.int32),
        outdated_depths=True,
    )


def init_network(
    inputs: chex.Array,
    output_size: int,
    max_nodes: int,
    key: chex.PRNGKey,
    scale_weights: float = 0.1,
) -> tuple[Network, ActivationState]:
    """Creates a Network and ActivationState from an input array."""

    input_size = len(inputs)
    n_initial_connections = input_size * output_size
    sender_receiver_pairs = cartesian_product(
        jnp.arange(input_size),
        jnp.arange(input_size, input_size + output_size),
        size=max_nodes,
        fill_value=-max_nodes,
    )
    senders = sender_receiver_pairs[:, 0]
    receivers = sender_receiver_pairs[:, 1]

    weights_init = (
        jax.random.normal(key, (n_initial_connections,), dtype=jnp.float32)
        * scale_weights
    )
    weights = jnp.zeros(max_nodes).at[:n_initial_connections].set(weights_init)

    node_types = jnp.concatenate(
        [
            jnp.zeros(input_size, dtype=jnp.int32),  # input nodes = 0
            jnp.full(output_size, 2, dtype=jnp.int32),  # output nodes = 2
            jnp.full(max_nodes - input_size - output_size, 3, dtype=jnp.int32),
        ]  # disabled nodes = 3
    )
    activation_fns = jnp.zeros(max_nodes, dtype=jnp.int32)  # no activation by default

    activations = get_initial_activations(inputs, senders)
    toggled_nodes = create_toggled_mask(node_types, senders, input_size)
    activation_counts = jnp.zeros(max_nodes, dtype=jnp.int32)
    activation_counts = jnp.zeros(max_nodes, dtype=jnp.int32)
    has_fired = jnp.zeros(max_nodes, dtype=jnp.int32)

    return (
        Network(
            node_indices=jnp.arange(max_nodes, dtype=jnp.int32),
            node_types=node_types,
            weights=weights,
            activation_fns=activation_fns,
            senders=senders,
            receivers=receivers,
            input_size=input_size,
            output_size=output_size,
            max_nodes=max_nodes,
        ),
        ActivationState(
            values=activations,
            toggled=toggled_nodes,
            activation_counts=activation_counts,
            has_fired=has_fired,
            node_depths=jnp.zeros(max_nodes, dtype=jnp.int32),
            outdated_depths=True,
        ),
    )

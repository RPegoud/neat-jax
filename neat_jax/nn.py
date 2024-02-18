from functools import partial

import chex
import jax
import jax.numpy as jnp

from neat_jax import ActivationState, Network


def make_network(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    weights: jnp.ndarray,
    activation_indices: jnp.ndarray,
    inputs: jnp.ndarray,
    node_types: jnp.ndarray,
    max_nodes: int,
    output_size: int,
) -> tuple[ActivationState, Network]:
    """
    Initializes the network and activation state for NEAT processing.
    """

    chex.assert_trees_all_equal_shapes(senders, receivers, weights)
    chex.assert_trees_all_equal_shapes(activation_indices, node_types)

    senders = (
        (jnp.ones(max_nodes, dtype=jnp.int32) * -1).at[: len(senders)].set(senders)
    )
    receivers = (
        (jnp.ones(max_nodes, dtype=jnp.int32) * -1).at[: len(receivers)].set(receivers)
    )
    weights = (jnp.zeros(max_nodes, dtype=jnp.float32)).at[: len(weights)].set(weights)
    activation_indices = (
        (jnp.zeros(max_nodes, dtype=jnp.int32))
        .at[: len(activation_indices)]
        .set(activation_indices)
    )

    activations = jnp.zeros(max_nodes).at[: len(inputs)].set(inputs)
    activated_nodes = jnp.int32(activations > 0)
    activation_counts = jnp.zeros(max_nodes, dtype=jnp.int32)
    has_fired = jnp.zeros(max_nodes, dtype=jnp.int32)

    return (
        ActivationState(
            values=activations,
            toggled=activated_nodes,
            activation_counts=activation_counts,
            has_fired=has_fired,
        ),
        Network(
            node_indices=jnp.arange(max_nodes, dtype=jnp.int32),
            node_types=node_types,
            weights=weights,
            activation_indices=activation_indices,
            senders=senders,
            receivers=receivers,
            output_size=output_size,
        ),
    )


def get_required_activations(net: Network, size: int) -> jnp.ndarray:
    """Calculates the number of activations each node requires to fire.

    Iterates over the network's receivers to count the incoming connections for each node,
    determining how many signals a node must receive before it can activate.

    Returns:
        An array where each element represents the required number of activations for the
        corresponding node.
    """

    def _carry(required_activations: jnp.ndarray, receiver: int):
        return (
            jax.lax.cond(
                receiver == -1,
                lambda _: required_activations,  # bypass this step for non-receiver nodes
                lambda _: required_activations.at[receiver].add(1),
                operand=None,
            ),
            None,
        )

    required_activations, _ = jax.lax.scan(
        _carry, (jnp.zeros(size, dtype=jnp.int32)), net.receivers
    )
    return required_activations


def get_active_connections(
    activation_state: ActivationState,
    net: Network,
    max_nodes: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Identifies active connections based on the current activation state.

    Filters the senders and receivers based on whether the sender nodes are currently active,
    indicating which connections will transmit signals in this iteration.

    Returns:
        A tuple of arrays for active senders and their corresponding receivers.
    """

    active_senders_indices = jnp.where(
        activation_state.toggled[net.senders] > 0,
        size=max_nodes,
        fill_value=max_nodes + 1,  # fill with out-of-range values
    )[0]
    active_senders = jnp.take(
        net.senders,
        active_senders_indices,
        axis=0,
        fill_value=-1,  # out-of-range values are converted to -1
        # which effectively ignores senders and receivers that are currently not toggled
    )
    active_receivers = jnp.take(
        net.receivers, active_senders_indices, axis=0, fill_value=-1
    )

    return active_senders, active_receivers


def get_activation(activation_index: int, x: float) -> jnp.float32:
    """
    Given an index, selects an activation function and computes `activation(x)`.

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
    """
    return jax.lax.switch(
        activation_index,
        [
            lambda x: 1 / (1 + jnp.exp(-x)),  # sigmoid
            lambda x: jnp.divide(1, x),  # inverse
            lambda x: jnp.sinh(x) / jnp.cosh(x),  # hyperbolic cosine
            lambda x: jnp.float32(jnp.maximum(0, x)),  # relu
            lambda x: jnp.float32(jnp.abs(x)),  # absolute value
            lambda x: jnp.sin(x),  # sine
            lambda x: jnp.exp(jnp.square(-x)),  # gaussian
            lambda x: jnp.float32(jnp.sign(x)),  # step
        ],
        operand=x,
    )


def forward_toggled_nodes(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    activation_state: ActivationState,
    net: Network,
) -> ActivationState:
    """Updates the activation values and counts for receiver nodes.

    For each active connection, adds the sender's activation value to the receiver's
    current value, increments the receiver's activation count, and deactivates the sender node
    for the next step.

    Returns:
        Updated ActivationState reflecting the new activations and deactivations.
    """

    def _add_single_activation(activation_state: jnp.ndarray, x: tuple) -> jnp.ndarray:

        def _update_activation_state(val: tuple) -> ActivationState:
            """
            Adds the activation of a sender to a receiver's value and
            increments its activation count, then deactivates the sender node.

            Note: the deactivation of the sender nodes will only be effective at the
            end of the iteration (at the next step when computing which nodes should fire).
            """
            activation_state, sender, receiver, weight, activation_index = val
            values = activation_state.values
            activation_counts = activation_state.activation_counts

            # conditionally apply the current node's activation function if
            # it belongs to the hidden layers (node_type == 1)
            sender_value = values.at[sender].get()
            values = jax.lax.cond(
                net.node_types.at[sender].get() == 1,
                lambda _: values.at[sender].set(
                    get_activation(activation_index, x=sender_value)
                ),
                lambda _: values,
                operand=None,
            )

            values = values.at[receiver].add(values[sender] * weight)
            activation_counts = activation_counts.at[receiver].add(1)
            toggled = activation_state.toggled.at[sender].set(0)
            has_fired = activation_state.has_fired.at[sender].set(1)
            return (
                activation_state.replace(
                    values=values,
                    activation_counts=activation_counts,
                    toggled=toggled,
                    has_fired=has_fired,
                ),
                None,
            )

        def _bypass(val: tuple):
            """Bypasses the update for a given node."""
            activation_state, *_ = val
            return (activation_state, None)

        sender, receiver, weight, activation_index = x

        # nodes with activation -1 are not enabled and should not fire
        return jax.lax.cond(
            sender == -1,
            _bypass,
            _update_activation_state,
            operand=(activation_state, sender, receiver, weight, activation_index),
        )

    activation_state, _ = jax.lax.scan(
        _add_single_activation,
        activation_state,
        jnp.stack((senders, receivers, net.weights, net.activation_indices), axis=1),
    )
    return activation_state


def toggle_receivers(
    activation_state: ActivationState,
    net: Network,
    max_nodes: int,
) -> ActivationState:
    """Determines which nodes will be activated in the next iteration.

    Compares the current activation counts against the required activations to identify
    nodes that have received enough signals to fire.

    Returns:
        ActivationState with the `toggled` field updated to reflect nodes that will activate next.
    """

    def _update_toggle(val: tuple) -> jnp.ndarray:
        """
        Returns a mask designating nodes to activate at the next step.
        A node will be activated at next step if:
            * It has not fired previously
            * (and) It has received values from all its senders
        TODO: positive activation count and has not fired are probably redundant
        try keeping only has not fired
        """
        activation_state, required_activations = val
        positive_activation_counts = jnp.int32(activation_state.activation_counts > 0)
        has_req_activations = activation_state.activation_counts == required_activations
        has_not_fired = jnp.invert(activation_state.has_fired)
        return positive_activation_counts & has_req_activations & has_not_fired

    def _terminate(val: tuple) -> jnp.ndarray:
        """Disables all nodes, ending of the forward pass."""
        return jnp.zeros(max_nodes, dtype=jnp.int32)

    required_activations = get_required_activations(net, max_nodes)
    done = jnp.all(activation_state.activation_counts == required_activations)

    activated_nodes = jax.lax.cond(
        done,
        _terminate,
        _update_toggle,
        operand=(activation_state, required_activations),
    )

    return activation_state.replace(
        toggled=activated_nodes,
    )


@partial(jax.jit, static_argnames=("max_nodes", "output_size"))
def forward(
    activation_state: ActivationState,
    net: Network,
    max_nodes: int,
    output_size: int,
    activate_final: bool = False,
) -> ActivationState:
    """Executes a forward pass through the NEAT network.

    Repeatedly processes activations based on the current state, updating node activations
    and toggles until no nodes are left to toggle.

    Args:
        max_nodes: The maximum number of nodes in the network, used to ensure consistent
        array sizes.

    Returns:
        The final ActivationState after processing all activations.
    """

    def _termination_condition(val: tuple) -> bool:
        """Iterate while some nodes are still toggled."""
        activation_state, _ = val
        return jnp.sum(activation_state.toggled) > 0

    def _body_fn(val: tuple):
        activation_state, net = val
        senders, receivers = get_active_connections(activation_state, net, max_nodes)
        activation_state = forward_toggled_nodes(
            senders, receivers, activation_state, net
        )
        activation_state = toggle_receivers(activation_state, net, max_nodes)

        return activation_state, net

    activation_state, net = jax.lax.while_loop(
        _termination_condition, _body_fn, (activation_state, net)
    )

    output_nodes_indices = jnp.where(net.node_types == 2, size=output_size)[0]
    activation_functions_indices = net.activation_indices.at[output_nodes_indices].get()
    outputs = activation_state.values.at[output_nodes_indices].get()

    outputs = jax.lax.cond(
        activate_final,
        lambda _: jax.vmap(get_activation)(activation_functions_indices, outputs),
        lambda _: outputs,
        operand=None,
    )

    return activation_state, outputs

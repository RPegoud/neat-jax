from functools import partial

import chex
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from .activation_fns import get_activation_fn
from .initialization import activation_state_from_inputs
from .neat_dataclasses import ActivationState, Network


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
                receiver < 0,
                lambda _: required_activations,  # bypass this step for non-receiver nodes
                lambda _: required_activations.at[receiver].add(1),
                operand=None,
            ),
            None,
        )

    required_activations, _ = jax.lax.scan(
        _carry,
        (jnp.zeros(size, dtype=jnp.int32)),
        net.receivers,  # TODO: replace init by jnp.zeros_like(net.senders,dtype=jnp.int32) and remove size arg
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
        """Conditionally applies the update function if the current node is enabled."""

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

            # if the current node is set to fire and is a hidden node (node_type==1),
            # apply the activation function
            sender_value = values.at[sender].get()
            values = jax.lax.cond(
                net.node_types.at[sender].get() == 1,
                lambda _: values.at[sender].set(
                    get_activation_fn(activation_index, x=sender_value)
                ),
                lambda _: values,
                operand=None,
            )

            # propagate the sender's value to the receiver
            values = values.at[receiver].add(values[sender] * weight)

            # update activation_state
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

        # nodes with negative indices are disabled and should not fire
        return jax.lax.cond(
            sender < 0,
            _bypass,
            _update_activation_state,
            operand=(
                activation_state,
                jnp.int32(sender),
                jnp.int32(receiver),
                weight,
                jnp.int32(activation_index),
            ),
        )

    activation_state, _ = jax.lax.scan(
        _add_single_activation,
        activation_state,
        jnp.stack((senders, receivers, net.weights, net.activation_fns), axis=1),
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

        A node will be activated at the next step if:
            * It has not fired previously
            * (and) It has received values from all its senders
        """
        activation_state, required_activations = val
        has_req_activations = activation_state.activation_counts == required_activations
        has_not_fired = jnp.invert(activation_state.has_fired)
        return has_req_activations & has_not_fired

    def _terminate(val: tuple) -> jnp.ndarray:
        """Disables all nodes, ending the forward pass."""
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


@partial(jax.jit, static_argnames=("config"))
def forward(
    inputs: chex.Array,
    net: Network,
    config: DictConfig,
    activate_final: bool = False,
) -> tuple[ActivationState, jnp.array]:
    """Executes a forward pass through the NEAT network.

    Repeatedly processes activations based on the current state, updating node activations
    and toggles until no nodes are left to toggle.

    Args:
        max_nodes: The maximum number of nodes in the network, used to ensure consistent
        array sizes.

    Returns:
        * ActivationState: The final ActivationState after processing all activations.
        * jnp.ndarray: The network's outputs
    """

    def _termination_condition(val: tuple) -> bool:
        """Iterate while some nodes are still toggled."""
        activation_state, _ = val
        return jnp.sum(activation_state.toggled) > 0

    def _body_fn(val: tuple):
        activation_state, net = val
        senders, receivers = get_active_connections(
            activation_state, net, config.network.max_nodes
        )
        activation_state = forward_toggled_nodes(
            senders, receivers, activation_state, net
        )
        activation_state = toggle_receivers(
            activation_state, net, config.network.max_nodes
        )

        return activation_state, net

    activation_state = activation_state_from_inputs(
        inputs, net.senders, net.node_types, config.input_size, config.network.max_nodes
    )

    activation_state, net = jax.lax.while_loop(
        _termination_condition, _body_fn, (activation_state, net)
    )

    output_nodes_indices = jnp.where(net.node_types == 2, size=config.output_size)[0]
    activation_functions_indices = net.activation_fns.at[output_nodes_indices].get()
    outputs = activation_state.values.at[output_nodes_indices].get()

    outputs = jax.lax.cond(
        activate_final,
        lambda _: jax.vmap(get_activation_fn)(activation_functions_indices, outputs),
        lambda _: outputs,
        operand=None,
    )

    return activation_state, outputs


def forward_single_depth(senders, receivers, activation_state):
    """
    Updates the depth of each node in the network based on the current activation state.

    Iterates through each active connection and updates the depth of receiver nodes
    based on the depth of their sender nodes.

    Args:
        * senders (jnp.ndarray): An array of sender node indices.
        * receivers (jnp.ndarray): An array of receiver node indices corresponding to senders.
        * activation_state (ActivationState): The current state of the network activations,
            including node depths.

    Returns:
        ActivationState: The updated activation state with potentially modified node depths.
    """

    def _add_single_depth(activation_state: jnp.ndarray, x: tuple) -> jnp.ndarray:
        """Conditionally applies the update function if the current node is enabled."""

        def _update_depth(val: tuple) -> ActivationState:
            activation_state, sender, receiver = val
            values = activation_state.values
            activation_counts = activation_state.activation_counts
            node_depths = activation_state.node_depths

            sender_depth = node_depths.at[sender].get()
            receiver_depth = node_depths.at[receiver].get()
            node_depths = node_depths.at[receiver].set(
                jnp.max(jnp.array([sender_depth + 1, receiver_depth]))
            )

            activation_counts = activation_counts.at[receiver].add(1)
            toggled = activation_state.toggled.at[sender].set(0)
            has_fired = activation_state.has_fired.at[sender].set(1)
            return (
                activation_state.replace(
                    values=values,
                    activation_counts=activation_counts,
                    toggled=toggled,
                    has_fired=has_fired,
                    node_depths=node_depths,
                ),
                None,
            )

        def _bypass(val: tuple):
            """Bypasses the update for a given node."""
            activation_state, *_ = val
            return (activation_state, None)

        sender, receiver = x

        # nodes with negative indices are disabled and should not fire
        return jax.lax.cond(
            sender < 0,
            _bypass,
            _update_depth,
            operand=(activation_state, jnp.int32(sender), jnp.int32(receiver)),
        )

    activation_state, _ = jax.lax.scan(
        _add_single_depth,
        activation_state,
        jnp.stack((senders, receivers), axis=1),
    )
    return activation_state


@partial(jax.jit, static_argnames=("max_nodes"))
def update_depth(
    activation_state: ActivationState,
    net: Network,
    max_nodes: int,
) -> tuple[ActivationState]:
    """
    Computes the depth of each node in the network by performing a forward pass simulation.

    Args:
        activation_state (ActivationState): The initial state of the network activations.
        net (Network): The network structure containing sender and receiver connections.
        max_nodes (int): The total number of nodes in the network, used for array sizes.

    Returns:
        ActivationState: The updated activation state with the final computed node depths.
    """

    def _termination_condition(val: tuple) -> bool:
        """Iterate while some nodes are still toggled."""
        activation_state, _ = val
        return jnp.sum(activation_state.toggled) > 0

    def _body_fn(val: tuple):
        activation_state, net = val
        senders, receivers = get_active_connections(activation_state, net, max_nodes)
        activation_state = forward_single_depth(senders, receivers, activation_state)
        activation_state = toggle_receivers(activation_state, net, max_nodes)

        return activation_state, net

    input_values = jnp.int32(net.node_types == 0) * activation_state.values
    activation_state = ActivationState.from_inputs(input_values, max_nodes)

    activation_state, net = jax.lax.while_loop(
        _termination_condition, _body_fn, (activation_state, net)
    )

    return activation_state.replace(outdated_depths=False)

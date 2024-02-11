from functools import partial

import jax
import jax.numpy as jnp

from neat_jax import ActivationState, Network


def init(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    inputs: jnp.ndarray,
    node_types: jnp.ndarray,
    max_nodes: int,
) -> tuple[ActivationState, Network]:

    senders = (
        (jnp.ones(max_nodes, dtype=jnp.int32) * -1).at[: len(senders)].set(senders)
    )
    receivers = (
        (jnp.ones(max_nodes, dtype=jnp.int32) * -1).at[: len(receivers)].set(receivers)
    )

    activations = jnp.zeros(max_nodes).at[: len(inputs)].set(inputs)
    activated_nodes = jnp.int32(activations > 0)
    activation_counts = jnp.zeros(max_nodes, dtype=jnp.int32)

    return (
        ActivationState(
            values=activations,
            toggled=activated_nodes,
            activation_counts=activation_counts,
        ),
        Network(
            node_indices=jnp.arange(max_nodes, dtype=jnp.int32),
            node_types=node_types,
            edges=jnp.ones(max_nodes),
            senders=senders,
            receivers=receivers,
        ),
    )


def get_required_activations(net: Network, size: int) -> jnp.ndarray:
    """
    Returns the required number of activations for each node to fire.
    """

    def _carry(required_activations: jnp.ndarray, receiver: int):
        return (
            jax.lax.cond(
                receiver == -1,
                lambda _: required_activations,
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

    active_senders_indices = jnp.where(
        activation_state.toggled[net.senders] > 0,
        size=max_nodes,
        fill_value=-1,
    )[0]
    active_senders = jnp.take(net.senders, active_senders_indices, axis=0)
    active_receivers = jnp.take(net.receivers, active_senders_indices, axis=0)

    return active_senders, active_receivers


def add_activations(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    activation_state: ActivationState,
) -> ActivationState:
    """
    For given sender nodes, iteratively computes the activation
    of receiver nodes while carrying the global activation state.
    """

    def _add_single_activation(activation_state: jnp.ndarray, x: tuple) -> jnp.ndarray:
        def _update_activation_state(val: tuple):
            """
            Adds the activation of a sender to a receiver's value and
            increments its activation count, then deactivates the sender node.

            Note: the deactivation of the sender nodes will only be effective at the
            end of the iteration (at the next step when computing which nodes should fire).
            """
            activation_state, sender, receiver = val
            values = activation_state.values
            activation_counts = activation_state.activation_counts

            values = values.at[receiver].add(values[sender])
            activation_counts = activation_counts.at[receiver].add(1)
            toggled = activation_state.toggled.at[sender].set(0)
            return (
                activation_state.replace(
                    values=values,
                    activation_counts=activation_counts,
                    toggled=toggled,
                ),
                None,
            )

        def _bypass(val: tuple):
            """Bypasses the update for a given node."""
            activation_state, _, _ = val
            return (activation_state, None)

        sender, receiver = x

        # nodes with activation -1 are not enabled and should not fire
        return jax.lax.cond(
            sender == -1,
            _bypass,
            _update_activation_state,
            operand=(activation_state, sender, receiver),
        )

    activation_state, _ = jax.lax.scan(
        _add_single_activation,
        activation_state,
        jnp.stack((senders, receivers), axis=1),
    )
    return activation_state


def toggle_receivers(
    activation_state: ActivationState,
    net: Network,
    max_nodes: int,
) -> ActivationState:
    """
    Returns an array of size ``max_neurons`` indicating which nodes have received
    all necessary activations and should fire at the next step.
    """

    def _update_toggle(val: tuple) -> jnp.ndarray:
        """Returns a mask designating nodes to activate at the next step."""
        activation_state, required_activations = val
        positive_activation_counts = jnp.int32(activation_state.activation_counts > 0)
        fully_activated = activation_state.activation_counts == required_activations
        return positive_activation_counts & fully_activated

    def _terminate(val: tuple) -> jnp.ndarray:
        """Disables all nodes, leading to termination of the forward pass."""
        return jnp.zeros(max_nodes, dtype=jnp.int32)

    required_activations = get_required_activations(net, max_nodes)
    done = jnp.all(activation_state.activation_counts == required_activations)

    activated_nodes = jax.lax.cond(
        done,
        _terminate,
        _update_toggle,
        operand=(activation_state, required_activations),
    )

    return activation_state.replace(toggled=activated_nodes)


@partial(jax.jit, static_argnames="max_nodes")
def forward(
    activation_state: ActivationState, net: Network, max_nodes
) -> ActivationState:
    """
    Computes a forward pass through an arbitrary Neat Network.
    """

    def _termination_fn(val: tuple) -> bool:
        """Iterate while some nodes are still toggled."""
        activation_state, _ = val
        return jnp.sum(activation_state.toggled) > 0

    def _body_fn(val: tuple):
        activation_state, net = val
        senders, receivers = get_active_connections(activation_state, net, max_nodes)
        activation_state = add_activations(senders, receivers, activation_state)
        activation_state = toggle_receivers(activation_state, net, max_nodes)

        return activation_state, net

    activation_state, net = jax.lax.while_loop(
        _termination_fn, _body_fn, (activation_state, net)
    )

    return activation_state

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import random

from .neat_dataclasses import ActivationState, Network
from .nn import update_depth


@struct.dataclass
class Mutations:
    max_nodes: int
    weight_shift_rate: jnp.float32 = 0.9
    weight_mutation_rate: jnp.float32 = 0.1
    add_node_rate: jnp.float32 = 0.03
    add_connection_rate: jnp.float32 = 0.05

    @partial(jax.jit, static_argnames=("self", "scale_weights"))
    def weight_shift(
        self, key: random.PRNGKey, net: Network, scale_weights: float = 0.1
    ) -> Network:
        """
        Shifts the network's weights by a small value sampled from the normal distribution.

        Args:
            net (Network): The network to shift weights from
            key (random.PRNGKey): The random key used to sample from the normal distribution
            scale (float): A scaling factor applied to the sampled values. By default, ``0.1``
            adjusts the standard deviation of the distribution to be ``0.01``

        Returns:
            Network: The network with updated weights
        """

        def _single_shift(
            weight: jnp.float32,
            epsilon: jnp.float32,
            mutate: jnp.bool_,
        ) -> jnp.float32:
            """
            Updates a single weight with a probability of `weight_shift_rate`.
            Unitialized weights (i.e. with a value of 0) are ignored.
            """

            def _shift(val: tuple):
                weight, epsilon = val
                return weight + epsilon

            def _bypass(val: tuple):
                weight, _ = val
                return jnp.float32(weight)

            return jax.lax.cond(
                jnp.logical_and(mutate, weight != 0),
                _shift,
                _bypass,
                operand=(weight, epsilon),
            )

        eps_key, mutate_key = random.split(key, num=2)
        epsilons = random.normal(eps_key, shape=(self.max_nodes,)) * scale_weights
        mutate_i = (
            random.uniform(mutate_key, shape=(self.max_nodes,)) < self.weight_shift_rate
        )
        shifted_weights = jax.vmap(_single_shift)(net.weights, epsilons, mutate_i)
        return net.replace(weights=shifted_weights)

    @partial(jax.jit, static_argnums=(0))
    def weight_mutation(
        self, key: random.PRNGKey, net: Network, scale_weights: float = 0.1
    ) -> Network:
        """
        Randomly mutates connections from the network by sampling new weights from the normal distribution.

        Args:
            net (Network): The network to mutate
            key (random.PRNGKey): The random key used to sample from the normal distribution
            scale (float): A scaling factor applied to the sampled values. By default, ``0.1``
            adjusts the standard deviation of the distribution to be ``0.01``

        Returns:
            Network: The network with updated weights
        """

        def _single_mutation(
            weight: jnp.float32,
            new_value: jnp.float32,
            mutate: jnp.bool_,
        ) -> jnp.float32:
            """
            Updates a single weight with a probability of `weight_mutation_rate`.
            Unitialized weights (i.e. with a value of 0) are ignored.
            """

            def _mutate(val: tuple):
                _, new_value = val
                return new_value

            def _bypass(val: tuple):
                weight, _ = val
                return jnp.float32(weight)

            return jax.lax.cond(
                jnp.logical_and(mutate, weight != 0),
                _mutate,
                _bypass,
                operand=(weight, new_value),
            )

        sample_key, mutate_key = random.split(key, num=2)
        new_values = random.normal(sample_key, shape=(self.max_nodes,)) * scale_weights
        mutate_i = (
            random.uniform(mutate_key, shape=(self.max_nodes,))
            < self.weight_mutation_rate
        )
        mutated_weights = jax.vmap(_single_mutation)(net.weights, new_values, mutate_i)
        return net.replace(weights=mutated_weights)

    @partial(jax.jit, static_argnames=("self", "scale_weights"))
    def add_node(
        self,
        key: random.PRNGKey,
        net: Network,
        scale_weights: float = 0.1,
    ) -> Network:
        """
        Inserts a new node in the network by splitting an existing connection.
        In practice, an existing `sender -> receiver` connection is replaced by
        `sender -> new_node` & `new_node -> receiver`.

        Args:
            net (Network): The network to mutate

            key (random.PRNGKey): Random key used to:
                ```
                * Sample the connection to split
                * Sample weights for new sub connections
                * Sample an activation function for the new node
                ```
            scale_weight (float, optional): The constant used to scale weights sampled from the
            normal distribution

        Returns:
            Network: The mutated network with updated fields
        """

        def _mutate_fn(
            net: Network, key: random.PRNGKey, scale_weights: float
        ) -> Network:
            node_key, weight_key, activation_key = random.split(key, num=3)
            new_node_index = net.n_enabled_nodes + 1

            # sample a connection to modify
            valid_senders = jnp.int32(net.node_types < 2)  # input and hidden nodes
            selected = random.choice(
                node_key,
                jnp.arange(self.max_nodes) * valid_senders,
                p=valid_senders / valid_senders.sum(),
            )

            selected_sender = net.senders[selected]
            selected_receiver = net.receivers[selected]

            # disable old sender -> receiver connection
            senders = net.senders.at[selected].mul(-1)
            receivers = net.receivers.at[selected].mul(-1)

            # determine where to append the new connection
            connection_pos = jnp.int32(
                jnp.min(
                    jnp.where(
                        net.senders == -self.max_nodes,
                        size=self.max_nodes,
                        fill_value=jnp.inf,
                    )[0]
                )
            )

            # add sender -> new_node and new_node -> receiver connections
            senders = senders.at[connection_pos].set(selected_sender)
            senders = senders.at[connection_pos + 1].set(new_node_index)
            receivers = receivers.at[connection_pos].set(new_node_index)
            receivers = receivers.at[connection_pos + 1].set(selected_receiver)

            # declare new node as hidden node (1)
            node_types = net.node_types.at[new_node_index].set(1)

            # initialize new weights, disable old sender -> receiver connection
            new_weights = random.normal(weight_key, shape=(2,)) * scale_weights
            weights = net.weights.at[
                jnp.array([connection_pos, connection_pos + 1])
            ].set(new_weights)
            weights = weights.at[selected].set(0)

            # select an activation function for the new node
            activation_index = random.randint(
                activation_key, shape=(), minval=0, maxval=8
            )
            activation_indices = net.activation_indices.at[connection_pos].set(
                activation_index
            )

            return net.replace(
                senders=senders,
                receivers=receivers,
                weights=weights,
                node_types=node_types,
                activation_indices=activation_indices,
            )

        def _bypass_fn(net: Network, *args) -> Network:
            """Bypasses the mutation function depending on the `mutate` flag."""
            return net

        can_add_node = (
            sum(net.node_types == 3) >= 2
        )  # we need at least two uninitialized node
        mutate = random.uniform(key) < self.add_node_rate
        net = jax.lax.cond(
            jnp.logical_and(mutate, can_add_node),
            lambda _: _mutate_fn(net, key, scale_weights),
            lambda _: _bypass_fn(net, key, scale_weights),
            operand=None,
        )

        return net

    @partial(jax.jit, static_argnames=("self", "max_nodes"))
    def add_connection(
        self,
        key: random.PRNGKey,
        net: Network,
        activation_state: ActivationState,
        max_nodes: int,
        scale_weights: float = 0.1,
    ) -> tuple[Network, ActivationState]:
        """
        Samples a new connection between nodes, ensuring it adheres to network topology order
        (i.e. the depth of the sender is lower than the depth of the receiver).

        Args:
            key (jax.random.PRNGKey): A PRNG key used for random operations.
            net (Network): The current state of the network, containing node types and connections.
            activation_state (ActivationState): The current activation state of the network, including node depths.
            max_nodes (int): The maximum number of nodes in the network, used for defining array sizes.

        Returns:
            Network: The mutated network with a new connection
            ActivationState: The activation state of the network with updated node depths
        """

        def _mutate_fn(
            key: random.PRNGKey,
            net: Network,
            activation_state: ActivationState,
            max_nodes: int,
        ) -> tuple[jnp.int32, jnp.int32, ActivationState]:
            """
            Samples a new connection to be added to the network, ensuring it complies with
            topological order.
            """
            node_key, receiver_key = random.split(key, num=2)

            # connections can only be added to input and hidden nodes
            valid_senders = jnp.int32(net.node_types < 2)
            selected_sender = random.choice(
                node_key,
                jnp.arange(max_nodes),
                p=valid_senders / valid_senders.sum(),  # uniform sampling
            )

            # conditionally compute node depths if current values are outdated
            activation_state = jax.lax.cond(
                activation_state.outdated_depths,
                lambda _: update_depth(activation_state, net, max_nodes),
                lambda _: activation_state,
                operand=(None),
            )
            node_depths = activation_state.node_depths
            selected_depth = node_depths.at[selected_sender].get()

            # receivers that are already linked with the selected sender
            existing_receivers = net.receivers * jnp.int32(
                net.senders == selected_sender
            )
            compatible_receivers = jnp.int32(node_depths > selected_depth) * jnp.arange(
                max_nodes
            )

            # receiver indices with higher depths than selected sender and no prior connection
            receiver_candidates = jnp.setdiff1d(
                compatible_receivers, existing_receivers, size=max_nodes
            )
            receiver_candidates_mask = jnp.sign(receiver_candidates)

            selected_receiver = random.choice(
                receiver_key,
                receiver_candidates,
                p=(
                    receiver_candidates_mask / receiver_candidates_mask.sum()
                ),  # uniform sampling
            )

            return selected_sender, selected_receiver, activation_state

        def _bypass_mutate_fn(*args):
            return (
                jnp.int32(0),
                jnp.int32(0),
                activation_state,
            )

        def _apply_mutation_fn(
            key: random.PRNGKey,
            selected_sender: jnp.ndarray,
            selected_receiver: jnp.ndarray,
            net: Network,
            scale_weights: float = 0.1,
        ) -> Network:
            """
            Adds the sampled connection to the network topology.

            This function is bypassed if the sampled receiver is equal to 0 (i.e., if no valid
            index could be sampled).
            """
            # determine where to append the new connection
            connection_pos = jnp.int32(
                jnp.min(
                    jnp.where(
                        net.senders == -self.max_nodes,
                        size=self.max_nodes,
                        fill_value=jnp.inf,
                    )[0]
                )
            )

            # add new connection
            senders = net.senders.at[connection_pos].set(selected_sender)
            receivers = net.receivers.at[connection_pos].set(selected_receiver)

            # initialize new weights, disable old sender -> receiver connection
            new_weight = random.normal(key) * scale_weights
            weights = net.weights.at[jnp.array([connection_pos])].set(new_weight)

            return net.replace(
                senders=senders,
                receivers=receivers,
                weights=weights,
            )

        def _bypass_apply_mutation(*args):
            return net

        mutate = random.uniform(key) < self.add_connection_rate
        selected_sender, selected_receiver, activation_state = jax.lax.cond(
            mutate,
            lambda _: _mutate_fn(key, net, activation_state, max_nodes),
            lambda _: _bypass_mutate_fn(),
            operand=None,
        )

        net = jax.lax.cond(
            selected_receiver > 0,  # if no receiver was sampled
            lambda _: _apply_mutation_fn(
                key, selected_sender, selected_receiver, net, scale_weights
            ),
            lambda _: _bypass_apply_mutation(),
            operand=None,
        )

        return net, activation_state

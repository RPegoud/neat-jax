import jax
import jax.numpy as jnp
from flax import struct
from jax import random

from .neat_dataclasses import ActivationState, Network


@struct.dataclass
class Mutations:
    max_nodes: int
    weight_shift_rate: jnp.float32 = 0.9
    weight_mutation_rate: jnp.float32 = 0.1
    add_node_rate: jnp.float32 = 0.03
    add_connection_rate: jnp.float32 = 0.05

    def weight_shift(
        self, net: Network, key: random.PRNGKey, scale: float = 0.1
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
        epsilons = random.normal(eps_key, shape=(self.max_nodes,)) * scale
        mutate_i = jnp.bool_(
            random.choice(mutate_key, jnp.array([0, 1]), shape=(self.max_nodes,))
        )
        shifted_weights = jax.vmap(_single_shift)(net.weights, epsilons, mutate_i)
        return net.replace(weights=shifted_weights)

    def weight_mutation(self, net: Network, key: random.PRNGKey, scale: float = 0.1):
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
        new_values = random.normal(sample_key, shape=(self.max_nodes,)) * scale
        mutate_i = jnp.bool_(
            random.choice(mutate_key, jnp.array([0, 1]), shape=(self.max_nodes,))
        )
        mutated_weights = jax.vmap(_single_mutation)(net.weights, new_values, mutate_i)
        return net.replace(weights=mutated_weights)

    def add_node(
        net: Network,
    ):
        raise NotImplementedError

    def add_connection(
        net: Network,
        activation_state: ActivationState,
    ):
        raise NotImplementedError

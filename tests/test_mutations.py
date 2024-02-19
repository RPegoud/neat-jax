import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jax import random

from neat_jax import Mutations, make_network

# TODO: add second topology and .5 mutation probability

topology_config_0 = {
    "max_nodes": 10,
    "senders": jnp.array([0, 1, 2, 4]),
    "receivers": jnp.array([4, 4, 3, 3]),
    "weights": jnp.array([1, 1, 1, 1]),
    "activation_indices": jnp.array([0, 0, 0, 0, 0]),
    "node_types": jnp.array([0, 0, 0, 2, 1]),
    "inputs": jnp.array([0.5, 0.8, 0.2]),
    "output_size": 1,
}
mutation_config_null = {
    "weight_shift_rate": 0.0,
    "weight_mutation_rate": 0.0,
    "add_node_rate": 0.0,
    "add_connection_rate": 0.0,
}
mutation_config_certain = {
    "weight_shift_rate": 1.0,
    "weight_mutation_rate": 1.0,
    "add_node_rate": 1.0,
    "add_connection_rate": 1.0,
}


class MutationTests(chex.TestCase, parameterized.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        (
            "topology_0_no_mutations",
            topology_config_0,
            mutation_config_null,
            {"seed": 0},
            {
                "shifted_weights": jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                "mutated_weights": jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                "added_node_weights": jnp.array(
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ),
                "added_nodes_senders": jnp.concatenate(
                    [
                        jnp.array([0, 1, 2, 4]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 6),
                    ]
                ),
                "added_nodes_receivers": jnp.concatenate(
                    [
                        jnp.array([4, 4, 3, 3]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 6),
                    ]
                ),
                "added_nodes_node_types": jnp.array([0, 0, 0, 2, 1, 3, 3, 3, 3, 3]),
            },
        ),
        (
            "topology_0_all_mutations",
            topology_config_0,
            mutation_config_certain,
            {"seed": 0},
            {
                "shifted_weights": jnp.array(
                    [0.7389442, 1.0033853, 1.1086333, 0.8519701, 0, 0, 0, 0, 0, 0]
                ),
                "mutated_weights": jnp.array(
                    [-0.26105583, 0.00338528, 0.10863334, -0.1480299, 0, 0, 0, 0, 0, 0]
                ),
                "added_node_weights": jnp.array(
                    [1.0, 0.0, 1.0, 1.0, 0.12636864, -0.00423024, 0.0, 0.0, 0.0, 0.0]
                ),
                "added_nodes_senders": jnp.concatenate(
                    [
                        jnp.array([0, -1, 2, 4, 1, 5]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 4),
                    ]
                ),
                "added_nodes_receivers": jnp.concatenate(
                    [
                        jnp.array([4, -4, 3, 3, 5, 4]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 4),
                    ]
                ),
                "added_nodes_node_types": jnp.array([0, 0, 0, 2, 1, 1, 3, 3, 3, 3]),
            },
        ),
    )
    def test_mutate(
        self, t_params: dict, n_params: dict, rng_params: int, expected: dict
    ):
        activation_state, net = make_network(**t_params)
        mutations = Mutations(max_nodes=t_params["max_nodes"], **n_params)
        key = random.PRNGKey(rng_params["seed"])

        shifted_weights = self.variant(mutations.weight_shift)(net, key).weights

        mutated_weights = self.variant(mutations.weight_mutation)(net, key).weights
        added_node_network = mutations.add_node(net, key)

        chex.assert_trees_all_close(
            shifted_weights, expected["shifted_weights"], atol=1e-4
        )
        chex.assert_trees_all_close(
            mutated_weights, expected["mutated_weights"], atol=1e-4
        )
        chex.assert_trees_all_equal(
            added_node_network.senders, expected["added_nodes_senders"]
        )
        chex.assert_trees_all_equal(
            added_node_network.receivers, expected["added_nodes_receivers"]
        )
        chex.assert_trees_all_equal(
            added_node_network.node_types, expected["added_nodes_node_types"]
        )

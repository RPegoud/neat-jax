import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jax import random

from neat_jax import Mutations, make_network

# TODO: add second topology

topology_config_0 = {
    "max_nodes": 10,
    "senders": jnp.array([0, 1, 2, 4]),
    "receivers": jnp.array([4, 4, 3, 3]),
    "weights": jnp.array([1, 1, 1, 1]),
    "activation_fns": jnp.array([0, 0, 0, 0, 0]),
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
mutation_config_0_5 = {
    "weight_shift_rate": 0.5,
    "weight_mutation_rate": 0.5,
    "add_node_rate": 0.5,
    "add_connection_rate": 0.5,
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
                "added_connection_weights": jnp.array(
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ),
                "added_connection_senders": jnp.concatenate(
                    [
                        jnp.array([0, 1, 2, 4]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 6),
                    ]
                ),
                "added_connection_receivers": jnp.concatenate(
                    [
                        jnp.array([4, 4, 3, 3]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 6),
                    ]
                ),
                "added_connection_node_types": jnp.array(
                    [0, 0, 0, 2, 1, 3, 3, 3, 3, 3]
                ),
                "outdated_depths": jnp.bool_(
                    True
                ),  # no mutation => depths were not updated
            },
        ),
        (
            "topology_0__0.5_mutations",
            topology_config_0,
            mutation_config_0_5,
            {"seed": 1},
            {
                "shifted_weights": jnp.array(
                    [0.9356227, 1.0, 0.9701904, 1.0, 0, 0, 0, 0, 0, 0]
                ),
                "mutated_weights": jnp.array(
                    [-0.06437729, 1.0, -0.02980961, 1.0, 0, 0, 0, 0, 0, 0]
                ),
                "added_node_weights": jnp.array(
                    [1.0, 1.0, 0.0, 1.0, -0.00611208, -0.10968596, 0.0, 0.0, 0.0, 0.0]
                ),
                "added_nodes_senders": jnp.concatenate(
                    [
                        jnp.array([0, 1, -2, 4, 2, 5]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 4),
                    ]
                ),
                "added_nodes_receivers": jnp.concatenate(
                    [
                        jnp.array([4, 4, -3, 3, 5, 3]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 4),
                    ]
                ),
                "added_nodes_node_types": jnp.array([0, 0, 0, 2, 1, 1, 3, 3, 3, 3]),
                "added_connection_weights": jnp.array(
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ),
                "added_connection_senders": jnp.concatenate(
                    [
                        jnp.array([0, 1, 2, 4]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 6),
                    ]
                ),
                "added_connection_receivers": jnp.concatenate(
                    [
                        jnp.array([4, 4, 3, 3]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 6),
                    ]
                ),
                "added_connection_node_types": jnp.array(
                    [0, 0, 0, 2, 1, 3, 3, 3, 3, 3]
                ),
                "outdated_depths": jnp.bool_(False),
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
                    [0.0, 1.0, 1.0, 1.0, 0.12636864, -0.00423024, 0.0, 0.0, 0.0, 0.0]
                ),
                "added_nodes_senders": jnp.concatenate(
                    [
                        jnp.array([0, 1, 2, 4, 0, 5]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 4),
                    ]
                ),
                "added_nodes_receivers": jnp.concatenate(
                    [
                        jnp.array([-4, 4, 3, 3, 5, 4]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 4),
                    ]
                ),
                "added_nodes_node_types": jnp.array([0, 0, 0, 2, 1, 1, 3, 3, 3, 3]),
                "added_connection_weights": jnp.array(
                    [1.0, 1.0, 1.0, 1.0, -0.02058423, 0.0, 0.0, 0.0, 0.0]
                ),
                "added_connection_senders": jnp.concatenate(
                    [
                        jnp.array([0, 1, 2, 4, 1]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 5),
                    ]
                ),
                "added_connection_receivers": jnp.concatenate(
                    [
                        jnp.array([4, 4, 3, 3, 3]),
                        jnp.repeat(jnp.array([-topology_config_0["max_nodes"]]), 5),
                    ]
                ),
                "added_connection_node_types": jnp.array(
                    [0, 0, 0, 2, 1, 3, 3, 3, 3, 3]
                ),
                "outdated_depths": jnp.bool_(False),
            },
        ),
    )
    def test_mutate(
        self, t_params: dict, n_params: dict, rng_params: int, expected: dict
    ):
        activation_state, net = make_network(**t_params)
        mutations = Mutations(max_nodes=t_params["max_nodes"], **n_params)
        key = random.PRNGKey(rng_params["seed"])

        # --- Independent mutations ---
        shifted_weights = self.variant(mutations.weight_shift)(key, net, 0.1).weights
        mutated_weights = self.variant(mutations.weight_mutation)(key, net, 0.1).weights
        added_node_network, added_node_activation_state = self.variant(
            mutations.add_node
        )(key, net, 0.1)
        added_connection_network, added_connection_activation_state = self.variant(
            mutations.add_connection, static_argnames=["self"]
        )(key, net, activation_state)

        # Shift weights tests
        chex.assert_trees_all_close(shifted_weights, expected["shifted_weights"])

        # Mutate weights tests
        chex.assert_trees_all_close(mutated_weights, expected["mutated_weights"])

        # Add node tests
        chex.assert_trees_all_close(
            added_node_network.weights, expected["added_node_weights"]
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

        # Add connection tests
        chex.assert_trees_all_equal(
            added_connection_network.senders, expected["added_connection_senders"]
        )
        chex.assert_trees_all_equal(
            added_connection_network.receivers, expected["added_connection_receivers"]
        )
        chex.assert_trees_all_equal(
            added_connection_network.node_types, expected["added_connection_node_types"]
        )
        assert (
            added_connection_activation_state.outdated_depths
            == expected["outdated_depths"]
        )

        # --- Sequential mutations ---

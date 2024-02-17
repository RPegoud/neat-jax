import chex
import jax.numpy as jnp
from absl.testing import parameterized

from neat_jax import init


class NetworkTests(chex.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "topology_0",
            {
                "max_nodes": 20,
                "senders": jnp.array([0, 1, 2, 4]),
                "receivers": jnp.array([4, 4, 3, 3]),
                "weights": jnp.array([1, 1, 1, 1]),
                "activation_indices": jnp.array([0, 0, 0, 0]),
                "node_types": jnp.array([0, 0, 0, 2, 1]),
                "inputs": jnp.array([0.5, 0.8, 0.2]),
                "output_size": 1,
            },
            {
                "n_nodes": 20,
                "type_counts": jnp.array([3, 1, 1]),
                "n_inputs": 3,
            },
        ),
        (
            "topology_1",
            {
                "max_nodes": 30,
                "senders": jnp.array([0, 1, 1, 2, 2, 4, 5, 5]),
                "receivers": jnp.array([4, 4, 5, 3, 5, 5, 3, 6]),
                "weights": jnp.array([1, 1, 1, 1, 1, 1, 1]),
                "activation_indices": jnp.array([0, 0, 0, 0, 0, 0, 0]),
                "node_types": jnp.array([0, 0, 0, 2, 1, 1, 2]),
                "inputs": jnp.array([0.4, 0.3, 0.5]),
                "output_size": 2,
            },
            {
                "n_nodes": 30,
                "type_counts": jnp.array([3, 2, 2]),
                "n_inputs": 3,
            },
        ),
    )
    def test_init(self, params: dict, expected: dict):
        activation_state, net = init(**params)

        chex.assert_trees_all_equal_shapes(
            net.node_indices, net.receivers, net.senders, net.weights
        )
        assert net.n_nodes == expected["n_nodes"]
        chex.assert_trees_all_equal(net.node_types_counts, expected["type_counts"])

        assert jnp.all(activation_state.activation_counts == 0)
        assert jnp.sum(activation_state.toggled) == expected["n_inputs"]
        assert jnp.sum(activation_state.values) == jnp.sum(params["inputs"])

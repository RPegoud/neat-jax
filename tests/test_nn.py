import chex
import jax.numpy as jnp
from absl.testing import parameterized

from neat_jax import forward, get_depth, get_required_activations, make_network

# TODO: test different activation functions within the network

topology_config_0 = {
    "max_nodes": 20,
    "senders": jnp.array([0, 1, 2, 4]),
    "receivers": jnp.array([4, 4, 3, 3]),
    "weights": jnp.array([1, 1, 1, 1]),
    "activation_indices": jnp.array([0, 0, 0, 0, 0]),
    "node_types": jnp.array([0, 0, 0, 2, 1]),
    "inputs": jnp.array([0.5, 0.8, 0.2]),
    "output_size": 1,
}

topology_config_1 = {
    "max_nodes": 30,
    "senders": jnp.array([0, 1, 1, 2, 2, 4, 5, 5]),
    "receivers": jnp.array([4, 4, 5, 3, 5, 5, 3, 6]),
    "weights": jnp.array([1, 1, 1, 1, 1, 1, 1, 1]),
    "activation_indices": jnp.array([0, 0, 0, 0, 0, 0, 0]),
    "node_types": jnp.array([0, 0, 0, 2, 1, 1, 2]),
    "inputs": jnp.array([0.4, 0.3, 0.5]),
    "output_size": 2,
}


class NetworkTests(chex.TestCase, parameterized.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        (
            "topology_0",
            topology_config_0,
            {
                "n_nodes": 20,
                "type_counts": jnp.array([3, 1, 1, 15]),
                "n_inputs": 3,
                "output_wo_activation": jnp.array([0.985835]),
                "output_w_activation": jnp.array([0.7282645]),
                "node_depths": jnp.append(jnp.array([0, 0, 0, 2, 1]), jnp.full(15, 0)),
            },
        ),
        (
            "topology_1",
            topology_config_1,
            {
                "n_nodes": 30,
                "type_counts": jnp.array([3, 2, 2, 23]),
                "n_inputs": 3,
                "output_wo_activation": jnp.array([1.3127818, 0.69270194]),
                "output_w_activation": jnp.array([0.7879783, 0.6665677]),
                "node_depths": jnp.append(
                    jnp.array([0, 0, 0, 3, 1, 2, 3, 0]), jnp.full(22, 0)
                ),
            },
        ),
    )
    def test_nn(self, params: dict, expected: dict):
        activation_state, net = make_network(**params)
        chex.assert_equal(activation_state.outdated_depths, jnp.bool_(True))

        # --- Test init ---
        chex.assert_trees_all_equal_shapes(
            net.node_indices, net.receivers, net.senders, net.weights
        )
        assert net.n_nodes == expected["n_nodes"]
        chex.assert_trees_all_equal(net.node_type_counts, expected["type_counts"])

        assert jnp.all(activation_state.activation_counts == 0)
        assert jnp.sum(activation_state.toggled) == expected["n_inputs"]
        assert jnp.sum(activation_state.values) == jnp.sum(params["inputs"])

        # --- Test forward ---
        activation_state, outputs_wo_activation = self.variant(
            forward, static_argnames=("max_nodes", "output_size")
        )(
            activation_state,
            net,
            params["max_nodes"],
            params["output_size"],
            activate_final=False,
        )

        chex.assert_trees_all_equal(
            activation_state.activation_counts,
            get_required_activations(net, params["max_nodes"]),
        )
        chex.assert_trees_all_equal(
            activation_state.toggled, jnp.zeros(params["max_nodes"])
        )
        chex.assert_trees_all_close(
            outputs_wo_activation, expected["output_wo_activation"]
        )

        activation_state, outputs_w_activation = self.variant(
            forward, static_argnames=("max_nodes", "output_size")
        )(
            activation_state,
            net,
            params["max_nodes"],
            params["output_size"],
            activate_final=True,
        )
        chex.assert_trees_all_close(
            outputs_w_activation, expected["output_w_activation"]
        )

        # --- test depth computation ---
        chex.assert_equal(activation_state.outdated_depths, jnp.bool_(True))
        activation_state = self.variant(get_depth, static_argnames=("max_nodes"))(
            activation_state, net, params["max_nodes"]
        )
        chex.assert_trees_all_equal(
            activation_state.node_depths, expected["node_depths"]
        )
        chex.assert_equal(activation_state.outdated_depths, jnp.bool_(False))

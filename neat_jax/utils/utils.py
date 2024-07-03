import chex
import jax.numpy as jnp
import jax.random as random
import networkx as nx
from jax.typing import DTypeLike

from ..neat_dataclasses import Network


def cartesian_product(
    arr1: chex.Array,
    arr2: chex.Array,
    size: int = None,
    fill_value: int = None,
    dtype: DTypeLike = jnp.int32,
):
    cartesian_prod = jnp.dstack(jnp.meshgrid(arr1, arr2)).reshape(-1, 2)

    if size is None:
        return cartesian_prod

    chex.assert_axis_dimension_lt(cartesian_prod, 0, size)
    resized_cartesian_prod = jnp.full(
        shape=(size, 2), fill_value=fill_value, dtype=dtype
    )
    return resized_cartesian_prod.at[: len(cartesian_prod)].set(cartesian_prod)


def sample_from_mask(
    key: random.PRNGKey,
    mask: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Samples an index uniformly given a masked array.
    """
    return random.choice(key, indices * mask, p=mask / mask.sum())


def plot_network(net) -> None:
    def convert_jraph_to_networkx_graph(net: Network) -> nx.Graph:
        n_nodes = len(net.node_indices)
        n_edges = len(net.weights)

        nx_graph = nx.DiGraph()
        if net.node_indices is None:
            for n in range(n_nodes[0]):
                nx_graph.add_node(n)
        else:
            for n in range(n_nodes):
                nx_graph.add_node(n, node_feature=net.node_indices[n])
        if net.weights is None:
            for e in range(n_edges):
                nx_graph.add_edge(int(net.senders[e]), int(net.receivers[e]))
        else:
            for e in range(n_edges):
                nx_graph.add_edge(
                    int(net.senders[e]),
                    int(net.receivers[e]),
                    edge_feature=net.weights[e],
                )
        return nx_graph

    nx_graph = convert_jraph_to_networkx_graph(net)
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos=pos, with_labels=True, node_size=200, font_color="yellow")

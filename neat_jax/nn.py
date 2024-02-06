import jax
import jax.numpy as jnp

from neat_jax import ConnectionGenes


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


@jax.jit
def forward_pass(
    connection_genes: ConnectionGenes, activations: jnp.array
) -> jnp.float32:
    """
    Processes the forward pass for a neural network defined by connection genes.
    The function iterates over each connection gene and updates the activations of
    the network nodes based on the connection weights and whether the connection is enabled.

    Args:
        connection_genes (ConnectionGenes): The named tuple containing arrays of the connection genes' attributes.
            It includes in-node indices, out-node indices, weights, enabled status, and innovation numbers.
        activations (jnp.array): An array containing the activation values of the network nodes. The size should
            correspond to the total number of nodes in the network.

    Returns:
        jnp.float32: The activation value of the last node in the network (i.e. the output node).

    The function first sorts the connection genes in topological order to respect the feedforward architecture.
    It then uses a `fori_loop` to iterate through the sorted connections, updating the activations array based
    on the enabled connections and their weights. The activation of each out-node is the sum of its incoming
    weighted activations, passed through a sigmoid activation function.
    """

    def _carry(idx, val):
        """Computes and update the activation for a single connection."""
        connection_genes, activations = val

        # select the attributes of the current connection gene
        in_idx, out_idx, weight, enabled, _ = jax.tree_map(
            lambda x: x[idx], connection_genes
        )

        activations = jax.lax.cond(
            enabled,
            lambda a: a.at[out_idx].add(sigmoid(activations[in_idx] * weight)),
            lambda a: a,
            operand=activations,
        )
        return connection_genes, activations

    # sorting genes in topological order
    # TODO: revise the function after mutations are implemented
    sorted_indices = jnp.argsort(connection_genes.out_node)
    sorted_attributes = jax.tree_map(
        lambda arr: arr[sorted_indices], [attr for attr in connection_genes]
    )
    sorted_genes = ConnectionGenes(*sorted_attributes)

    connection_genes, activations = jax.lax.fori_loop(
        0, len(sorted_genes.in_node), _carry, (sorted_genes, activations)
    )

    return activations[-1]

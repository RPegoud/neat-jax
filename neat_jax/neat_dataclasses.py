from typing import NamedTuple

import jax.numpy as jnp


class ConnectionGenes(NamedTuple):
    """
    Represents the genes that define a connection between two nodes in a NEAT network.

    Attributes:
        in_node (jnp.int32): The index of the input node for this connection.
        out_node (jnp.int32): The index of the output node for this connection.
        weight (jnp.float32): The weight of this connection, which scales the input from the in_node.
        enabled (jnp.bool_): A boolean indicating whether the connection is active (enabled) or not.
        innovation_number (jnp.int32): A unique identifier for the historical marking of this gene.

    The innovation number helps in identifying corresponding genes across different genomes
    during the crossover operation and is crucial for tracking the evolution of topologies.
    """

    in_node: jnp.int32
    out_node: jnp.int32
    weight: jnp.float32
    enabled: jnp.bool_
    innovation_number: jnp.int32


class NodeGenes(NamedTuple):
    """
    Represents the genes that define a node in a NEAT network.

    Attributes:
        type (str): The type of the node, which can be 'input', 'hidden', or 'output'.
        index (jnp.float32): The unique index or identifier for this node within the network.

    The index is used to refer to this node in connection genes. The type of the node affects
    how it is processed during the forward pass of the network; for instance, input nodes might
    be handled differently from hidden and output nodes.
    """

    type: str
    index: jnp.int32

import jax

from neat_jax import Mutations, forward, init_network

config = {
    "key": jax.random.key(0),
    "input_size": 3,
    "output_size": 2,
    "max_nodes": 10,
}
mutation_config_certain = {
    "weight_shift_rate": 1,
    "weight_mutation_rate": 1,
    "add_node_rate": 1,
    "add_connection_rate": 1,
}


def main(config: dict):
    inputs = jax.random.normal(config["key"], (config["input_size"],))

    net, activation_state = init_network(
        inputs, config["output_size"], config["max_nodes"], config["key"]
    )
    mutations = Mutations(max_nodes=config["max_nodes"], **mutation_config_certain)
    net = mutations.add_node(config["key"], net)
    activation_state, y = forward(
        inputs, net, config["max_nodes"], config["input_size"], config["output_size"]
    )

    print(activation_state, y)


if __name__ == "__main__":
    main(config)

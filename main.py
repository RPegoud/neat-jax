import hydra
import jax
from omegaconf import DictConfig, OmegaConf

from neat_jax import Mutations, forward, init_network, log_config

INPUT_SIZE = 4
OUTPUT_SIZE = 3


def run_exp(config: dict):
    config.input_size = INPUT_SIZE
    config.output_size = OUTPUT_SIZE

    key = jax.random.key(config.params.seed)
    inputs = jax.random.normal(key, (config.input_size,))

    net, activation_state = init_network(
        inputs, config.output_size, config.network.max_nodes, key
    )
    mutations = Mutations(max_nodes=config.network.max_nodes, **config.mutations)
    net = mutations.add_node(key, net)
    activation_state, y = forward(inputs, net, config)

    print(activation_state, y)


@hydra.main(
    config_path="neat_jax/configs",
    config_name="default_config.yaml",
    version_base="1.3.2",
)
def hydra_entry_point(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    log_config(cfg)
    return run_exp(cfg)


if __name__ == "__main__":
    hydra_entry_point()

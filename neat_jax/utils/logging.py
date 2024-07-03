import json

from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf


def log_config(config: DictConfig) -> None:
    config = OmegaConf.to_object(config)
    print(f"{Fore.BLUE}{Style.BRIGHT}Running Neat experiment:")
    print(
        f"{Fore.GREEN}{Style.BRIGHT}Hyperparameters:"
        f"{Style.NORMAL}{json.dumps(config, sort_keys=True, indent=4)}{Style.RESET_ALL}"
    )

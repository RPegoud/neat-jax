import json

from colorama import Fore, Style


def log_config(data: dict) -> None:
    print(f"{Fore.BLUE}{Style.BRIGHT}Running Neat experiment:")
    print(
        f"{Fore.GREEN}{Style.BRIGHT}Hyperparameters:"
        f"{Style.NORMAL}{json.dumps(data, sort_keys=True, indent=4)}{Style.RESET_ALL}"
    )

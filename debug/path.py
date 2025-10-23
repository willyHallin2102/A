#!/usr/bin/env python3
"""
    debug/path.py
    -------------
    CLI for Channel Modelling with VAE
"""

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np

from database.loader import DataLoader, shuffle_and_split
from src.models.chanmod import ChannelModel
from logs.logger import LogLevel, Logger, get_loglevel
from src.config.const import PREPROC_FN



def parse_args() -> argparse.Namespace:
    """
    Create and parse the CLI arguments, these arguments are functionalities that 
    enable interactions with the path model to the channel model, and by
    extension whichever generative model attached.

    Returns:
    --------
        argparse.Namespace, the parsed arguments.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Channel modelling CLI -- Train and generate channel parameters"
    )

    # Adding global and common arguments for each subparser.
    parser.add_argument(
        "--log-level", type=str, default="INFO", 
        help="Logging level (debug, info, warning, error, critical)"
    )
    parser.add_argument(
        "--force-overwrite", action="store_true",
        help="If set, overwrite existing model without prompting"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for training."
    )
    parser.add_argument(
        "--ratio", type=float, default=0.1, help="train data ratio in [0.00, 1.00]"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seeding for reproducibility"
    )

    #; Main subcommands, each assign a individual sub-parser ;#
    sub = parser.add_subparsers(
        dest="command", required=False, help="Command to execute"
    )

    train_parser = sub.add_parser("train", help="Train channel model")
    train_parser.add_argument(
        "--city", type=str, default="beijing", 
        help="City or comma-separated list of cities or `all`"
    )
    train_parser.add_argument(
        "--model-type", type=str, default="vae", help="model type e.g., vae"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="#train-loops"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate for the optimizer, that being during backpropagation"
    )


    return parser.parse_args()


def check_existing_model(directory: Path) -> bool:
    return (directory / PREPROC_FN).exists()


def prompt_user() -> bool:
    while True:
        response = input("Model already exists. Overwrite? [y/N]").strip().lower()
        if response in ['y', 'yes']: return True
        elif response in ['n', 'no', '']: return False
        else: print("Please answer 'y' for yes or 'n' for no.")


def get_data_files(city_arg: str) -> list[Path]:
    """
    Resolve which data files to train on.
    """
    supported = {"beijing", "boston", "london", "moscow", "tokyo"}
    if city_arg.strip().lower() == "all":
        return [Path(f"uav_{city}/train.csv") for city in sorted(supported)]

    cities = {city.strip().lower() for city in city_arg.split(",")}
    invalid = cities - supported
    if invalid:
        raise ValueError(f"Unsupported cities: {', '.join(sorted(invalid))}. "
                         f"Supported: {', '.join(sorted(supported))}")
    return [Path(f"uav_{city}/train.csv") for city in sorted(cities)]



def main():
    """
    """
    args = parse_args()

    loglevel = get_loglevel(args.log_level)
    logger = Logger(name="path-debug", level=loglevel, to_disk=False)
    loader = DataLoader(level=loglevel)

    data = loader.load(get_data_files(args.city))
    dtr, dts = shuffle_and_split(
        data=data, val_ratio=args.ratio, seed=args.seed
    )
    

    model = ChannelModel(level=loglevel, model_type=args.model_type)
    # model.path.build()

    model_exists = check_existing_model(model.path.directory)
    load_existing = False
    if model_exists and not args.force_overwrite:
        if prompt_user():
            print("Loading existing model...")
            model.path.build()
        else:
            print("Overriding existing model...")
            model.path.load()
            load_existing = True
    elif model_exists and args.force_overwrite:
        print("Force overwrite enabled. Overwriting existing model...")
        model.path.build()
    else:
        print("No existing model found. Building new model...")
        model.path.build()


    history = model.path.fit(
        dtr=dtr, dts=dts, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    model.path.save()


if __name__ == "__main__":
    main()

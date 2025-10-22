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

from database.loader import DataLoader
from src.models.chanmod import ChannelModel
from logs.logger import LogLevel
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
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
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



    return parser.parse_args()


def main():
    """
    """
    args = parse_args()


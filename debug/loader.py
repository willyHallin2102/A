#!/usr/venv python3.10
"""
    database/loader.py
    ------------------
    Command-line interface for testing and debugging the `DataLoader`
    system. Includes timing different methods, loading UAV trajectory
    files, console commands, JSON logging the performances with the 
    various levels, `LogLevel`. Running basic integrity tests.
"""
from __future__ import annotations

import argparse
import os, sys

import numpy as np
import pandas as pd

from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

# Sleeker 
# sys.path.append(str(Path(__file__).parent))

from logs.logger import Logger, LogLevel
from database.loader import DataLoader, shuffle_and_split


# ---------------========== CLI Utilities ==========--------------- #

def time_function(
    f, *args, logger: Optional[Logger]=None, label=None, **kwargs
):
    """
    Measures and log execution time of any function
    """
    start = time.perf_counter()
    result = f(*args, **kwargs)
    elapsed = time.perf_counter() - start
    if logger:
        logger.info(f"[{label or f.__name__}] Took {elapsed:.6f}s")
    return result, elapsed


# ---------------========== Test Routines ==========--------------- #

def test_loader_functionality(args, logger):
    """
    """
    logger.info("===== Starting DataLoader Debug Test =====")
    loader = DataLoader(
        n_workers=args.workers, chunk_size=args.chunk_size,
        level=LogLevel.DEBUG if args.verbose else LogLevel.INFO
    )

    n = args.samples
    mock_data = {
        "dvec_x"    : np.random.randn(n),
        "dvec_y"    : np.random.randn(n),
        "dvec_z"    : np.random.randn(n),
        "rx_type"   : np.random.randint(0, 5, n),
        "link_state": np.random.randint(0, 2, n),
        "los_pl"    : np.random.rand(n),
        "los_ang"   : np.random.rand(n),
        "los_dly"   : np.random.rand(n),
        "nlos_pl"   : np.random.rand(n),
        "nlos_ang"  : np.random.rand(n),
        "nlos_dly"  : np.random.rand(n),
    }

    # Save step
    with logger.time_block("Saving data"):
        loader.save(mock_data, "mock_dataset", fmt="csv")

    # --- Load step (and measure different methods)
    with logger.time_block("Loading data"):
        processed = loader.load(["mock_dataset.csv"])

    with logger.time_block("Shuffle and split"):
        train, val = shuffle_and_split(processed, val_ratio=0.2)

    logger.info(f"Train set size: {len(next(iter(train.values())))}")
    logger.info(f"Val set size: {len(next(iter(val.values())))}")

    logger.info("=== DataLoader Debug Test Complete ===")



def compare_methods(args, logger):
    """Compare threading vs multiprocessing performance using logger."""
    logger.info("===== Comparing Thread vs Process Executor =====")

    loader = DataLoader(n_workers=args.workers, chunk_size=args.chunk_size,
                        level=LogLevel.DEBUG if args.verbose else LogLevel.INFO)

    test_files = ["mock_dataset.csv"] * args.repeats
    methods = ["thread", "process"]
    results = {}

    for method in methods:
        with logger.time_block(f"{method.upper()} Execution"):
            start = time.perf_counter()
            loader.load(test_files)
            elapsed = time.perf_counter() - start
            results[method] = elapsed
            logger.info(f"{method.upper()} took {elapsed:.4f}s")

    logger.info(f"Summary: {results}")
    faster = min(results, key=results.get)
    logger.info(f"Faster method: {faster.upper()}")

    logger.info("===== Comparison Done =====")


# ---------------========== CLI Argument Parser ==========--------------- #

def main():
    parser = argparse.ArgumentParser(
        description="CLI debug + performance testing for DataLoader."
    )
    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "compare"],
                        help="Run mode: 'test' for basic loader test, 'compare' for executor performance.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of workers (default: CPU count).")
    parser.add_argument("--chunk-size", type=int, default=10_000,
                        help="Chunk size for loading data.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeat count for comparison runs.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose debug logging.")
    parser.add_argument("--samples", type=int, default=100_000,
                        help="Number of rows randomly generate")
    args = parser.parse_args()

    # Initialize root logger
    logger = Logger("CLI_Debug", level=LogLevel.DEBUG if args.verbose else LogLevel.INFO)

    logger.info("CLI Debug Mode Initiated.")
    logger.info(f"Args: {args}")

    if args.mode == "test":
        test_loader_functionality(args, logger)
    elif args.mode == "compare":
        compare_methods(args, logger)

    logger.info("CLI Execution Completed.")
    Logger.shutdown_all()





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)

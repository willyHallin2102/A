"""
    debug/coords.py
    ---------------
    #!/usr/bin/venv python3
    Scripts in the debugging does not store any results in form of 
    logs, `to_disk=False` prevent storage.
"""
from __future__ import annotations

import os, sys
import argparse

import numpy as np
from pathlib import Path
from typing import Any, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.math import coords
from logs.logger import Logger, LogLevel



# ----------======= Generating Vectors =======---------- #

def generate_vector(args, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random 3D vector with reproducibility by applying a 
    assignable seed value.
    """
    vector = rng.normal(size=(args.n_samples, 3)).astype(args.dtype)
    vector /= np.linalg.norm(vector, axis=1, keepdims=True)
    vector *= rng.uniform(0.1, 10.0, size=(args.n_samples, 1))

    return vector


def generate_spherical(
    args, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    radius = rng.uniform(0.1, 10.0, size=args.n_samples).astype(args.dtype)
    phi = rng.uniform(-180.0, 180.0, size=args.n_samples).astype(args.dtype)
    theta = rng.uniform(0.0, 180.0, size=args.n_samples).astype(args.dtype)

    return radius, phi, theta


# ----------======= Test 1: Cartesian and Spherical =======---------- #

def test_cartesian_to_spherical(args, logger: Logger):
    rng = np.random.default_rng(args.seed)
    logger.info("cartesian-to-spherical started")
    vector = generate_vector(args, rng)
    with logger.time_block("cartesian-to-spherical"):
        _, _, _ = coords.cartesian_to_spherical(vector)
    logger.info("cartesian-to-spherical completed")


def test_spherical_to_spherical(args, logger: Logger):
    rng = np.random.default_rng(args.seed)
    logger.info("spherical-to-cartesian started")
    r, p, t = generate_spherical(args, rng)
    with logger.time_block("spherical-to-cartesian"):
        _ = coords.spherical_to_cartesian(r, p, t)
    logger.info("spherical-to-cartesian completed")


def test_roundtrip(args, logger: Logger):
    """ """
    if args.n_samples > 10_000:
        raise TimeoutError("This number of samples is unfeasible")
    rng = np.random.default_rng(args.seed)
    logger.info("roundtrip started")

    errors: List[float] = []
    for trial in range(args.trials):
        vector_1 = generate_vector(args, rng)

        r, phi, theta = coords.cartesian_to_spherical(vector_1)
        vector_2 = coords.spherical_to_cartesian(r, phi, theta)

        error = np.linalg.norm(vector_2-vector_1, axis=1)
        errors.append(error)

        # logger.info(f"[{trial+1}/{args.trials}] error: {error}")
    
    errors = np.concatenate(errors)
    logger.info(
        f"Max Error: {np.max(errors):.5e}\tMean Error: {np.mean(error):.5e}\n"
    )





# ----------======= Test 2: Rotations (Angle Combinations) =======---------- #

def _generate_angles(args, logger: Logger, rng):
    _, phi_0, theta_0 = generate_spherical(args, rng)
    _, phi_1, theta_1 = generate_spherical(args, rng)
    # logger.info("generated angles")

    return phi_0, theta_0, phi_1, theta_1


def test_add_angles(args, logger: Logger):
    """
        Adding angles 
    """
    rng = np.random.default_rng(args.seed)
    phi_0, theta_0, phi_1, theta_1 = _generate_angles(args, logger, rng)

    with logger.time_block("Add Angles"):
        phi, theta = coords.add_angles(phi_0, theta_0, phi_1, theta_1)
    logger.info("add-angles, completed")
    
def test_sub_angles(args, logger: Logger):
    """
        Subtracting angles 
    """
    rng = np.random.default_rng(args.seed)
    phi_0, theta_0, phi_1, theta_1 = _generate_angles(args, logger, rng)

    with logger.time_block("Sub Angles"):
        phi, theta = coords.sub_angles(phi_0, theta_0, phi_1, theta_1)
    logger.info("sub-angles, completed")




def test_rotation(args, logger):
    """
    Tests that add_angles and sub_angles are inverses of each other.
    """
    rng = np.random.default_rng(args.seed)
    errors: List[float] = []

    with logger.time_block("Rotation Consistency Test"):
        for trial in range(args.trials):
            phi_0, theta_0, phi_1, theta_1 = _generate_angles(args, logger, rng)

            phi_r, theta_r = coords.add_angles(phi_0, theta_0, phi_1, theta_1)
            phi_b, theta_b = coords.sub_angles(phi_r, theta_r, phi_1, theta_1)

            diff_phi = np.mod(phi_b - phi_0 + np.pi, 2 * np.pi) - np.pi
            diff_theta = theta_b - theta_0
            errors.append(np.sqrt(diff_phi**2 + diff_theta**2))

        mean_err = np.mean(errors)
        max_err = np.max(errors)

    logger.info(f"Rotation test completed over {args.trials} trials.")
    logger.info(f"Mean angular error: {mean_err:.6e}, Max error: {max_err:.6e}")





# ----------======= Test 1: Building CLI Parser =======---------- #

def build_parser() -> argparse.ArgumentParser:
    """
    """
    parser = argparse.ArgumentParser(description="CLI debug `debug.coords.py`")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser):
        p.add_argument("--n-samples", type=int, default=1000)
        p.add_argument("--repeat", type=int, default=3)
        p.add_argument("--dtype", type=lambda x: getattr(np, x), default=np.float64)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--debug", type=bool, default=False)
        p.add_argument("--trials", type=int, default=100)
    
    p1 = sub.add_parser("cart-to-sph", help="Cartesian -> spherical conversion")
    add_common(p1)

    p2 = sub.add_parser("sph-to-cart", help="Spherical -> Cartesian conversion")
    add_common(p2)

    p3 = sub.add_parser(
        "roundtrip", help="Measures cartesian -> spherical -> cartesian errors"
    )
    add_common(p3)

    p4 = sub.add_parser("rotate", help="Spherical angle rotation add/sub")
    add_common(p4)

    p5 = sub.add_parser("add-angles", help="Add Angles")
    add_common(p5)
    p6 = sub.add_parser("sub-angles", help="Subtract angles")
    add_common(p6)

    p7 = sub.add_parser("rotate", help="add and subtract the angles to measure errors")
    add_common(p7)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = Logger(
        "coords-cli", to_disk=False,
        level=LogLevel.DEBUG if args.debug else LogLevel.INFO
    )

    if args.command == "cart-to-sph":
        test_cartesian_to_spherical(args, logger)
    elif args.command == "sph-to-cart":
        test_spherical_to_spherical(args, logger)
    elif args.command == "roundtrip":
        test_roundtrip(args, logger)
    elif args.command == "rotate":
        test_rotation(args, logger)
    elif args.command == "add-angles":
        test_add_angles(args, logger)
    elif args.command == "sub-angles":
        test_sub_angles(args, logger)
    elif args.command == "rotate":
        test_rotation(args, logger)
    
    Logger.shutdown_all()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)

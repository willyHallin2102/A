#!/usr/bin/env python3
"""
    test_chanmod_cli.py
    ------------------------------------
    CLI test suite for `chanmod.py`.

    Usage:
        python test_chanmod_cli.py --all
        python test_chanmod_cli.py --quick
        python test_chanmod_cli.py --profile
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.mockup.array import UniformRectangularArray, multi_sector_array
from src.mockup.chanmod import (
    MultiPathMChannel,
    directional_path_loss,
    directional_path_loss_multi_sector,
)
from src.config.data import LinkState


# --------------------------- Utility Functions --------------------------- #

def make_random_channel(n_rays: int = 10, seed: int = 42) -> MultiPathMChannel:
    """Generate a random multipath channel instance for testing."""
    rng = np.random.default_rng(seed)
    channel = MultiPathMChannel()
    channel.link_state = LinkState.LOS if n_rays > 0 else LinkState.NO_LINK
    channel.path_loss = rng.uniform(60, 120, n_rays).astype(np.float32)
    channel.delays = rng.uniform(0, 300e-9, n_rays).astype(np.float32)
    channel.angles = rng.uniform(
        0, 180, (n_rays, MultiPathMChannel.N_ANGLES)
    ).astype(np.float32)
    return channel


def make_test_arrays(frequency: float = 28e9) -> tuple[
    UniformRectangularArray, UniformRectangularArray
]:
    """Create representative TX and RX arrays."""
    return (
        UniformRectangularArray(n_antenna=(4, 4), frequency=frequency),
        UniformRectangularArray(n_antenna=(4, 4), frequency=frequency),
    )


# --------------------------- Individual Tests --------------------------- #

def test_basic_path_loss() -> None:
    """Test basic directional path loss calculation."""
    print("---> Test: Basic directional path loss")

    tx, rx = make_test_arrays()
    channel = make_random_channel(8)

    result = directional_path_loss(
        tx, rx, channel,
        return_element_gain=True,
        return_beamforming_gain=True,
    )

    path_loss_effective, tx_element, rx_element, tx_bf, rx_bf = result

    assert np.isfinite(path_loss_effective), "Path loss not finite!"
    assert np.all(np.isfinite(tx_element)), "TX element gain invalid!"
    assert np.all(np.isfinite(rx_element)), "RX element gain invalid!"
    assert np.all(np.isfinite(tx_bf)) and np.all(np.isfinite(rx_bf)), "Beamforming gain invalid!"

    print(f"Passed — path_loss_effective = {path_loss_effective:.2f} dB")


def test_multi_sector() -> None:
    """Test multi-sector beamforming and best-array selection."""
    print("---> Test: Multi-sector directional path loss")

    tx_base, rx_base = make_test_arrays()
    tx_list = multi_sector_array(tx_base, sector_type="azimuth", n_sector=3)
    rx_list = multi_sector_array(rx_base, sector_type="azimuth", n_sector=3)
    channel = make_random_channel(12)

    result = directional_path_loss_multi_sector(tx_list, rx_list, channel)
    path_loss_effective, ind_tx, ind_rx = result[:3]

    assert np.isfinite(path_loss_effective), "path_loss_effective not finite!"
    assert isinstance(ind_tx, (int, np.integer)), "TX index invalid!"
    assert isinstance(ind_rx, (int, np.integer)), "RX index invalid!"

    print(
        f"Passed — Best TX={ind_tx}, RX={ind_rx}, "
        f"path_loss_effective={path_loss_effective:.2f} dB"
    )


def test_no_link() -> None:
    """Test behaviour for NO_LINK channels (should return large path loss)."""
    print("---> Test: No-link case")

    tx, rx = make_test_arrays()
    channel = make_random_channel(0)

    # Request scalar only to avoid tuple return
    path_loss_effective = directional_path_loss(tx, rx, channel, return_element_gain=False)

    assert np.isclose(
        path_loss_effective, MultiPathMChannel.PATH_LOSS
    ), f"Expected {MultiPathMChannel.PATH_LOSS}, got {path_loss_effective}"

    print(f"Passed — no link handled correctly (path_loss = {path_loss_effective:.1f} dB)")


def test_rms_delay() -> None:
    """Test RMS delay spread computation."""
    print("---> Test: RMS delay computation")

    channel = make_random_channel(5)
    rms = channel.compute_rms_delay()

    assert rms >= 0.0, "RMS delay must be non-negative!"
    print(f"Passed — RMS delay = {rms * 1e9:.3f} ns")


def test_omni_path_loss() -> None:
    """Test omnidirectional path loss computation."""
    print("---> Test: Omni-directional path loss computation")

    channel = make_random_channel(5)
    path_loss_omni = channel.compute_omni_path_loss()

    assert np.isfinite(path_loss_omni), "Omni path loss not finite!"
    print(f"Passed — Omni path loss = {path_loss_omni:.2f} dB")


def test_performance(n_runs: int = 1000) -> None:
    """Simple performance benchmark for directional_path_loss."""
    print("---> Performance benchmark")

    tx, rx = make_test_arrays()
    channel = make_random_channel(16)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        directional_path_loss(tx, rx, channel, return_element_gain=False)
    elapsed = (time.perf_counter() - t0) / n_runs * 1e6

    print(f"Average time per run: {elapsed:.2f} µs")


# --------------------------- CLI Entrypoint --------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive CLI tests for chanmod.py")
    parser.add_argument("--quick", action="store_true", help="Run a quick subset of tests")
    parser.add_argument("--profile", action="store_true", help="Run performance test only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")

    args = parser.parse_args()
    if not any([args.quick, args.profile, args.all]):
        args.all = True

    print("\n-----=== mmWave Channel Model Test Runner ===-----\n")

    try:
        if args.quick:
            test_basic_path_loss()
            test_no_link()
        elif args.profile:
            test_performance()
        else:
            test_basic_path_loss()
            test_multi_sector()
            test_no_link()
            test_rms_delay()
            test_omni_path_loss()
            test_performance()
    except AssertionError as e:
        print(f"\n[Error] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")
        sys.exit(2)
    else:
        print("\n[Success] All tests completed successfully.\n")


if __name__ == "__main__":
    main()

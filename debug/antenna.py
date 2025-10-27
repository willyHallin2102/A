"""
    debug/antenna.py
    ----------------
"""
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.mockup.antenna import (
    ElementIsotropic, Element3GPP, ArrayLike,
    plot_pattern, add_angles, sub_angles
)

# Later replace with logger
import traceback
import time


def test_isotropic_basic():
    print("Testing Isotropic Antenna")
    iso = ElementIsotropic()

    phi = np.linspace(-180, 180, 361)
    theta = np.linspace(-90, 90, 181)
    gains = iso.response(phi, theta)

    assert np.allclose(gains, 0.0), "Isotropic response not uniform"
    print("Passed isotropic test")


def test_3gpp_basic():
    print("Testing Element3GPP basic behaviour...")
    el = Element3GPP()
    phi = np.linspace(-180, 180, 361)
    theta = np.linspace(-90, 90, 181)
    phi_mat, theta_mat = np.meshgrid(phi, theta, indexing='xy')
    g = el.response(phi_mat, theta_mat)
    assert np.isfinite(g).all(), "3GPP response returned NaN/Inf values!"
    print("Passed 3GPP finite output test.")


def test_plot_modes(show_plot=True):
    print("Testing plot_pattern modes...")
    def mock_pattern(phi, theta):
        return np.cos(np.radians(phi)) * np.cos(np.radians(theta))
    
    modes = ['rect_phi', 'rect_theta', '2d', 'polar_phi', 'polar_theta']
    for mode in modes:
        try:
            phi, theta, v, ax, im = plot_pattern(
                mock_pattern, n_phi=50, n_theta=25, plot_type=mode
            )
            assert v.shape == (25, 50)
            print(f"{mode} ok!")
            if show_plot:
                plt.title(mode)
                plt.show(block=False)
                plt.pause(0.2)
                plt.close()
        except Exception:
            print(f"{mode} failed")
            traceback.print_exc()


def stress_test(n_iter=1000):
    print(f"→ Running stress test with {n_iter} random samples...")
    rng = np.random.default_rng(42)
    el = Element3GPP()
    t0 = time.time()
    for i in range(n_iter):
        phi = rng.uniform(-180, 180, 100)
        theta = rng.uniform(-90, 90, 100)
        g = el.response(phi, theta)
        assert np.isfinite(g).all(), f"NaN detected at iteration {i}"
    elapsed = time.time() - t0
    print(f"✓ Stress test complete in {elapsed:.2f}s ({n_iter} iterations)")


def main():
    parser = argparse.ArgumentParser(description="Antenna Module CLI Tester")
    parser.add_argument("--plot", action="store_true", help="Show radiation plots")
    parser.add_argument("--stress", type=int, default=0, help="Run stress test with N iterations")
    args = parser.parse_args()

    print("-----=== Antenna Module Test Runner ===-----")
    try:
        test_isotropic_basic()
        test_3gpp_basic()
        test_plot_modes(show_plot=args.plot)
        if args.stress > 0:
            stress_test(args.stress)
        print("[OK] All tests completed successfully.")
    except AssertionError as e:
        print("[FAIL] Assertion failed:", e)
        exit(1)
    except Exception:
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
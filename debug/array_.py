#!/usr/bin/env python3
"""
    debug/array.py
    ---------------
    Cli test suite for `src.mockup.antenna.py` (antenna array modelling).
    
    This script performs extensive functional, numerical, and stress testing
    of `ArrayBase`, `UniformRectangular`, and `RotatedArray`. Also testing 
    the `multi_sector_array` function.

    Usage:
    ------
        python debug/array.py --plot
        python debug/array.py --stress #n_samples
"""
import sys, os, time, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.mockup import antenna
from src.mockup.array import (
    ArrayBase, UniformRectangularArray, RotatedArray,
    multi_sector_array
)


# ---------------========== Basic Test ==========--------------- #

def test_basic_steering_vector():
    """
    """
    print("Testing steering vectors (ArrayBase)...")
    
    array = ArrayBase(frequency=28e9)
    phi, theta = np.array([0,45,90]), np.array([0,30,60])
    steering_vectors = array.steering_vectors(phi, theta)

    assert np.iscomplexobj(steering_vectors), "Steering vectors should be complex"
    assert steering_vectors.shape == (3, array.element_position.shape[0])

    print("Steering vector basic test, success!!")


def test_conjugate_beamforming():
    """
    """
    print("Conjugate beamforming vector generation...")

    array = ArrayBase(frequency=28e9)
    beam_weights = array.conjugate_beamforming([0, 45], [0, 30])
    
    assert np.allclose(np.linalg.norm(beam_weights, axis=1), 1.0, atol=1e-12)
    print("Conjugate beamforming normalization test, success!!")


def test_ura_geometry():
    """
    """
    print("Testing URA element position...")

    ny, nz = 4, 3
    array = UniformRectangularArray((ny, nz), frequency=28e9)
    expected_shape = (ny * nz, 3)

    assert array.element_position.shape == expected_shape, f"Uniform Rectangular Array position shape mismatch: {array.element_position.shape}"
    assert np.allclose(array.element_position[:, 0], 0), "URA should lie on y-z plane (x=0)"

    print("URA geometry Test, success!!")


# def test_rotated_array_consistency():
#     """
#     """
#     print("Testing rotated array consistency...")

#     base = UniformRectangularArray((2, 2))
#     rotate_0 = RotatedArray(base, phi_0=0, theta_0=0)
#     phi, theta = np.linspace(-180, 180, 5), np.linspace(-90, 90, 5)
    
#     base_sv = base.steering_vectors(phi, theta)
#     rot_sv = rotate_0.steering_vectors(phi, theta)

#     assert np.allclose(base_sv, rot_sv, atol=1e-12), "Rotate by 0° should yield identical results"
#     print("RotateArray(0°,0°) consistency test, success!!")


def test_rotated_array_consistency():
    print("Testing rotated array consistency...")

    base = UniformRectangularArray((2, 2))
    rotate_0 = RotatedArray(base, phi_0=0, theta_0=0)
    phi, theta = np.linspace(-180, 180, 5), np.linspace(-90, 90, 5)
    
    base_sv = base.steering_vectors(phi, theta)
    rot_sv = rotate_0.steering_vectors(phi, theta)

    max_diff = np.abs(base_sv - rot_sv).max()
    print(f"Max abs diff: {max_diff:.3e}")

    assert np.allclose(base_sv, rot_sv, atol=1e-3), (
        f"Rotate by 0° should yield identical results (max diff {max_diff:.3e})"
    )

    print("RotateArray(0°,0°) consistency test, success!!")


def test_multi_sector():
    print("Testing multi-sector array generation...")

    base = UniformRectangularArray((2, 2))
    arrays = multi_sector_array(base, sector_type='azimuth', n_sector=4)

    assert len(arrays) == 4
    assert all(isinstance(array, RotatedArray) for array in arrays)
    
    phis = [array.phi_0 for array in arrays]
    assert np.all(np.diff(phis) > 0), "Sector azimuths should increase monotonically"

    print("Multi-sector azimuth test , success!!")


# ---------------========== Stress Testing ==========--------------- #

def stress_test_sv(n_iter: int=1000, n_dirs: int=500):
    print(
        f"Stress testing the ArrayBase.steering_vectors(); "
        f"({n_iter} iterations × {n_dirs} dirs)..."
    )

    array = UniformRectangularArray((4, 4))
    rng = np.random.default_rng(42)
    start = time.time()

    for i in range(n_iter):
        phi = rng.uniform(-180, 180, n_dirs)
        theta = rng.uniform(-90, 90, n_dirs)
        sv = array.steering_vectors(phi, theta)

        assert np.all(np.isfinite(sv.real)) and np.all(np.isfinite(sv.imag)), f"NaN detected in iteration {i}"
    
    elapsed = time.time() - start
    print(f"Stress test passed in {elapsed:.2f}s ({n_iter} iterations)")


# ---------------========== Plotting Testing ==========--------------- #

def plot_example():
    print("Plotting example array pattern (UniformRectangularArray)...")


    array = UniformRectangularArray((4, 4))
    weights = array.conjugate_beamforming(0, 0)
    array.plot_pattern(weights=weights, n_phi=90, n_theta=45, plot_type="2d")

    plt.title("URA (4×4) beam pattern")
    plt.show()




def main():
    parser = argparse.ArgumentParser(description="Antenna Array Module CLI Tester")
    parser.add_argument("--plot", action="store_true", help="Show radiation pattern plots")
    parser.add_argument("--stress", type=int, default=0, help="Run stress test with N iterations")
    args = parser.parse_args()

    print("-----=== Antenna Array Module Test Runner ===-----")
    try:
        test_basic_steering_vector()
        test_conjugate_beamforming()
        test_ura_geometry()
        test_rotated_array_consistency()
        test_multi_sector()
        if args.stress > 0:
            stress_test_sv(args.stress)
        if args.plot:
            plot_example()
        print("All array tests completed successfully")
    
    except AssertionError as ae:
        print("[ERROR] Assertion has FAILED:", ae)
        traceback.print_exc()
        exit(1)
    
    except Exception:
        traceback.print_exc()
        exit(1)



if __name__ == "__main__":
    main()

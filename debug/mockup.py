"""
    debug/mockup.py
    ---------------
    Debug running script for the src/mockup utilities, used to 
    resemble antennas in simulations e.g., 3GPP antennas for wireless
    communications channels.
"""
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.mockup.antenna import ElementIsotropic, Element3GPP
from src.mockup.array import UniformRectangularArray, RotatedArray
from src.mockup.chanmod import MultiPathChannel, directional_path_loss
from src.config.data import LinkState


#### NOTE !!! Need to change print to Logger instance, fix this later...
def test_antenna(args):
    """
    """
    print("antenna test")

    if args.antenna_type == "isotropic":
        antenna = ElementIsotropic()
    elif args.antenna_type == "3gpp":
        antenna = Element3GPP(
            phi_beamwidth=args.phi_bw,
            theta_beamwidth=args.theta_bw
        )
    else:
        raise ValueError(f"Unknown antenna type: {args.antenna_type}")
    
    # Need to change to more randomized vectors
    phi = np.array([0, 45, 90, 135, 180])
    theta = np.array([0, 30, 60])

    gains = antenna.response(phi, theta)
    mean_gain = antenna.compute_mean_gain(n_samples=args.samples)

    print(f"Antenna type: {args.antenna_type}")
    print(f"Gains at test angles:\n{gains}")
    print(f"Mean gain: {mean_gain:.2f} dBi")

    if args.plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        antenna.plot_pattern(
            plot_type="rect_phi", ax=ax, n_phi=360, n_theta=1,
            label=f"{args.antenna_type} (BW: {args.phi_bw}x{args.theta_bw})"
        )
        ax.set_title(f"Antenna Pattern - {args.antenna_type}")
        ax.legend()
        plt.savefig(f'antenna_{args.antenna_type}.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot to antenna_{args.antenna_type}.png")




def test_array(args):
    """
    """
    print("Testing antenna arrays...")

    # Create antenna element
    if args.element_type == "isotropic":
        element = ElementIsotropic()
    else:
        element = Element3GPP(phi_beamwidth=65, theta_beamwidth=65)

    # Create Uniform Rectangular Array element
    ura = UniformRectangularArray(
        n_antennas=(args.ny, args.nz), element=element,
        frequency=args.frequency
    )

    print(f"URA: {args.ny}x{args.nz} elements")
    print(f"Frequency: {args.frequency/1e9:.1f} GHz")
    print(f"Element type: {args.element_type}")
    print(f"Total antennas: {ura.element_position.shape[0]}")

    # Test steering vectors
    if args.test_steering:
        phi = np.array([0, 30, 60])
        theta = np.array([0, 15, 30])
        steering_vec, element_gain = ura.steering_vectors(phi, theta, return_element_gain=True)
        print(f"Steering vectors shape: {steering_vec.shape}")
        print(f"Element gains: {element_gain}")
    
    # Test beamforming
    if args.test_beamforming:
        weights = ura.conjugate_beamforming(args.bf_phi, args.bf_theta)
        print(f"Beamforming weights shape: {weights.shape}")
        print(f"Beamforming towards: phi={args.bf_phi}, theta={args.bf_theta}")
        
        if args.plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ura.plot_pattern(
                weights=weights, plot_type="2d", ax=ax, include_element=True
            )
            ax.set_title(f"URA Pattern - {args.ny}x{args.nz}")
            plt.savefig(f'ura_{args.ny}x{args.nz}.png', dpi=150, bbox_inches='tight')
            print(f"Saved plot to ura_{args.ny}x{args.nz}.png")


def test_channel(args):
    """
    """
    print("Testing channel model...")
    
    # Create arrays
    element = Element3GPP(phi_beamwidth=65, theta_beamwidth=65)
    tx_array = UniformRectangularArray(
        n_antennas=(4, 4), element=element, frequency=args.frequency
    )

    rx_array = UniformRectangularArray(
        n_antennas=(2, 2), element=element, frequency=args.frequency
    )

    # Create channel with multipath
    channel = MultiPathChannel()
    channel.link_state = LinkState.LOS
    
    # Generate random multipath scenario
    np.random.seed(args.seed)
    n_paths = args.paths

    channel.path_loss = 80 + 20 * np.random.random(n_paths)  # 80-100 dB
    channel.delays = 10e-9 * (1 + 9 * np.random.random(n_paths))  # 10-100 ns
    channel.angles = np.column_stack([
        np.random.uniform(-180, 180, n_paths),  # aoa_phi
        np.random.uniform(0, 90, n_paths),      # aoa_theta  
        np.random.uniform(-180, 180, n_paths),  # aod_phi
        np.random.uniform(0, 90, n_paths)       # aod_theta
    ])

    print(f"Multipath scenario: {n_paths} paths")
    print(f"Path losses: {channel.path_loss}")
    print(f"Delays: {channel.delays}")
    print(f"RMS delay spread: {channel.rms_delays():.2e} s")
    
    # Test path loss calculations
    omni_pl = channel.compute_omni_path_loss()
    directional_pl = directional_path_loss(
        tx_array=tx_array,
        rx_array=rx_array,
        channel=channel,
        return_element_gain=False,
        return_beamforming_gain=False
    )
    
    print(f"Omni-directional path loss: {omni_pl:.2f} dB")
    print(f"Directional path loss: {directional_pl:.2f} dB")
    print(f"Beamforming gain: {omni_pl - directional_pl:.2f} dB")



def main():
    parser = argparse.ArgumentParser(description="Test antenna system modules")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Antenna test command
    antenna_parser = subparsers.add_parser('antenna', help='Test antenna elements')
    antenna_parser.add_argument('--antenna-type', choices=['isotropic', '3gpp'], 
                              default='3gpp', help='Antenna type')
    antenna_parser.add_argument('--phi-bw', type=float, default=65.0, 
                              help='Azimuth beamwidth (3GPP only)')
    antenna_parser.add_argument('--theta-bw', type=float, default=65.0, 
                              help='Elevation beamwidth (3GPP only)')
    antenna_parser.add_argument('--samples', type=int, default=1000,
                              help='Number of samples for mean gain calculation')
    antenna_parser.add_argument('--plot', action='store_true',
                              help='Generate plots')
    
    # Array test command  
    array_parser = subparsers.add_parser('array', help='Test antenna arrays')
    array_parser.add_argument('--ny', type=int, default=4, help='Y-dimension elements')
    array_parser.add_argument('--nz', type=int, default=4, help='Z-dimension elements')
    array_parser.add_argument('--frequency', type=float, default=28e9, 
                            help='Carrier frequency (Hz)')
    array_parser.add_argument('--element-type', choices=['isotropic', '3gpp'],
                            default='3gpp', help='Element type')
    array_parser.add_argument('--test-steering', action='store_true',
                            help='Test steering vectors')
    array_parser.add_argument('--test-beamforming', action='store_true',
                            help='Test beamforming')
    array_parser.add_argument('--bf-phi', type=float, default=0.0,
                            help='Beamforming azimuth angle')
    array_parser.add_argument('--bf-theta', type=float, default=0.0,
                            help='Beamforming elevation angle')
    array_parser.add_argument('--plot', action='store_true',
                            help='Generate plots')
    
    # Channel test command
    channel_parser = subparsers.add_parser('channel', help='Test channel model')
    channel_parser.add_argument('--paths', type=int, default=3,
                              help='Number of multi-paths')
    channel_parser.add_argument('--frequency', type=float, default=28e9,
                              help='Carrier frequency (Hz)')
    channel_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed')
    
    args = parser.parse_args()
    
    if args.command == 'antenna': test_antenna(args)
    elif args.command == 'array': test_array(args)
    elif args.command == 'channel': test_channel(args)
    else: parser.print_help()



if __name__ == "__main__":
    main()

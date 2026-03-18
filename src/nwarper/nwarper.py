from nwarper.diffusion_solve import diffusion_solve
import argparse
import warp as wp


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-solution",
        help="Plots the solution using vtk libraries",
        action="store_true",
    )
    parser.add_argument(
        "--plot-geometry",
        help="Plots the geometry using plotly then exits",
        action="store_true",
    )
    parser.add_argument(
        "input_file",
    )
    parser.add_argument(
        "--device",
        help=(
            "The default device that warp will use. "
            f"The list of available devices is {wp.get_devices()}"
        ),
        default="cpu",
    )
    args = parser.parse_args()

    diffusion_solve(**vars(args))

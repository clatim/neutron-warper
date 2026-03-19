from nwarper.diffusion_solve import diffusion_solve
from nwarper.reader import read_problem
from pathlib import Path
import numpy as np
import h5py
import pytest

inputs = Path(__file__).parent.resolve() / "inputs"


@pytest.mark.parametrize(
    "mesh",
    [
        "[[0.0, 1.0, 0.25]]",
        "[[0.0, 1.0, 0.1]]",
        "[[0.0, 1.0, 0.05]]",
    ],
)
def test_problem1(tmp_path, mesh):

    outfile = tmp_path / "problem1.h5"
    diffusion_solve(
        input_file=inputs / "problem1.yaml",
        output_file=outfile,
        device="cpu",
        overrides=[f"mesh.x.spacing={mesh}"],
    )

    with h5py.File(outfile, "r") as h:
        phi = np.array(h["flux"])

    settings, *_ = read_problem(inputs / "problem1.yaml")
    solution = np.ones_like(phi)
    np.testing.assert_allclose(phi, solution, rtol=settings.convergence_criteria)

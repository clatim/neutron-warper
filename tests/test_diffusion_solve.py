from nwarper.diffusion_solve import diffusion_solve
from nwarper.reader import read_problem
from pathlib import Path
import numpy as np

inputs = Path(__file__).parent.resolve() / "inputs"


def test_problem1():

    phi = diffusion_solve(input_file=inputs / "problem1.yaml", device="cpu")
    settings, *_ = read_problem(inputs / "problem1.yaml")
    solution = np.ones_like(phi)
    np.testing.assert_allclose(phi, solution, rtol=settings.convergence_criteria)

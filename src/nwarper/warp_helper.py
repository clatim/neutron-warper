import numpy as np
import warp as wp
import warp.fem as fem


def create_fem_field(
    geometry: fem.Geometry,
    field_values: np.array,
) -> fem.Field:
    """Creates a fem.Field as is needed by Warp of a np.array."""

    field_values = np.array(field_values)
    function_space = fem.make_polynomial_space(geometry, degree=0, discontinuous=True)
    field = function_space.make_field()
    field.dof_values = wp.from_numpy(field_values, dtype=float)

    return field


def setup_functionspace(geo: fem.Geometry, degree: int = 2):
    """wrapper for creating a function space"""

    functionspace = fem.make_polynomial_space(geo, degree=degree)
    return functionspace

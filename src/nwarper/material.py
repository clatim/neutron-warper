from dataclasses import dataclass
import warp.fem as fem
import numpy as np
import warp as wp
from nwarper.mixins import Printable


class MaterialConfiguration(Printable):
    def __init__(self, material_dict):

        self.names = list(material_dict.keys())

        self.materials = {}
        self.id_map = {}
        for i, (k, v) in enumerate(material_dict.items()):
            self.id_map[k] = i
            self.materials[i] = Material(**v)

    def id(self, name):
        return self.id_map[name]


@dataclass(kw_only=True)
class Material(Printable):
    sigt: float
    D: float  # The diffusion coefficient
    fixed_source: float = 0


def create_material_field(
    materials: MaterialConfiguration,
    geometry: fem.Geometry,
    property: str,
    material_ids: np.array,
) -> fem.Field:
    """Creates a fem.Field as is needed by Warp of a given property."""

    material_values = []
    for id in np.nditer(material_ids):
        material_values.append(getattr(materials.materials[int(id)], property))

    material_values = np.array(material_values)
    material_function_space = fem.make_polynomial_space(
        geometry, degree=0, discontinuous=True
    )
    material_field = material_function_space.make_field()
    material_field.dof_values = wp.from_numpy(material_values, dtype=float)

    return material_field


def build_material_field(mesh, regions):
    """Creates material field from input"""

    mat = np.zeros((mesh.nx, mesh.ny), dtype=int)

    for region in regions:
        mask = region.mask(mesh.Xc, mesh.Yc)

        mat[mask] = region.material_id

    return mat

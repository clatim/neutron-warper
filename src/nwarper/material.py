import warp.fem as fem
import numpy as np
from nwarper.mixins import Printable
from nwarper.warp_helper import create_fem_field


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


class Material(Printable):
    def __init__(self, sigt, sigf=0, sigs=0, D=None, fixed_source=0, chi=0, nu=0):
        self.sigt = sigt
        self.sigf = sigf
        self.sigs = sigs
        if D:
            self.D = D
        else:
            self.D = 1 / (3 * sigt)
        self.fixed_source = fixed_source
        self.chi = chi
        self.nu = nu

        self.sigr = sigt - sigs


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
    return create_fem_field(geometry, material_values)


def build_material_field(mesh, regions):
    """Creates material field from input"""

    mat = np.zeros((mesh.nx, mesh.ny), dtype=int)

    for region in regions:
        mask = region.mask(mesh.Xc, mesh.Yc)

        mat[mask] = region.material_id

    return mat

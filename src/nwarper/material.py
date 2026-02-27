from dataclasses import dataclass
import warp.fem as fem
import numpy as np
import warp as wp

class MaterialConfiguration():
    def __init__(self, material_dict):

        self.materials = {}
        for k, v in material_dict.items():
            self.materials[k] = Material(**v)

@dataclass(kw_only=True)
class Material():
    sigt: float
    D: float # The diffusion coefficient
    fixed_source: float = 0

def create_material_field(materials: MaterialConfiguration, geometry: fem.Geometry, property: str, material_ids) -> fem.Field:
    """ Creates a fem.Field as is needed by Warp of a given property.
    """

    material_values = []
    for id in material_ids:
        material_values.append(
            getattr(materials.materials[id], property)
        )

    material_values = np.array(material_values)
    material_function_space = fem.make_polynomial_space(geometry, degree=0, discontinuous=True)
    material_field = material_function_space.make_field()
    material_field.dof_values = wp.from_numpy(material_values, dtype=float)

    return material_field

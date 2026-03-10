from warp import fem
from nwarper.material import create_material_field


class CrossSectionFields:
    def __init__(self, geometry, materials, material_ids):

        self.D: fem.Field = create_material_field(
            materials, geometry, "D", material_ids
        )
        self.sigt: fem.Field = create_material_field(
            materials, geometry, "sigt", material_ids
        )
        self.sigs: fem.Field = create_material_field(
            materials, geometry, "sigt", material_ids
        )
        self.sigr: fem.Field = create_material_field(
            materials, geometry, "sigt", material_ids
        )
        self.fixed_source: fem.Field = create_material_field(
            materials,
            geometry,
            "fixed_source",
            material_ids,
        )
        self.sigf: fem.Field = create_material_field(
            materials, geometry, "sigf", material_ids
        )
        self.chi: fem.Field = create_material_field(
            materials, geometry, "chi", material_ids
        )
        self.nu: fem.Field = create_material_field(
            materials, geometry, "nu", material_ids
        )

from nwarper.structured_mesh import StructuredMesh
from omegaconf import OmegaConf
from nwarper.material import MaterialConfiguration, build_material_field
from nwarper.neutronproblem import NeutronProblem

from nwarper.config_schema import Config
from nwarper.region_builder import build_regions
from nwarper.settings import Settings


def read_problem(file):
    schema = OmegaConf.structured(Config)
    cfg = OmegaConf.load(file)
    OmegaConf.merge(schema, cfg)

    domain = NeutronProblem(
        xlen=cfg.domain.size[0],
        ylen=cfg.domain.size[1],
        xmin_alpha=cfg.domain.boundary_conditions.xmin,
        xmax_alpha=cfg.domain.boundary_conditions.xmax,
        ymin_alpha=cfg.domain.boundary_conditions.ymin,
        ymax_alpha=cfg.domain.boundary_conditions.ymax,
    )

    mesh = StructuredMesh.from_config(cfg.mesh)

    materials = MaterialConfiguration(cfg.materials)

    regions = build_regions(cfg.regions, materials)

    material_field = build_material_field(mesh, regions)

    settings = Settings(**cfg.settings)

    return settings, domain, mesh, materials, regions, material_field

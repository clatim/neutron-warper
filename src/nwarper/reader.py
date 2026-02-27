from omegaconf import OmegaConf
from nwarper.material import MaterialConfiguration
from nwarper.neutronproblem import NeutronProblem
from nwarper.mesh import Mesh
    
def read_problem(file):
    cfg = OmegaConf.load(file)
    problem_def = OmegaConf.to_container(cfg, resolve=True)
    mat_cfg = MaterialConfiguration(problem_def["materials"])
    problem = NeutronProblem(
        xlen=cfg.domain.xlen,
        ylen=cfg.domain.ylen,
        xmin_alpha=cfg.domain.boundary_conditions.xmin,
        xmax_alpha=cfg.domain.boundary_conditions.xmax,
        ymin_alpha=cfg.domain.boundary_conditions.ymin,
        ymax_alpha=cfg.domain.boundary_conditions.ymax,
    )

    mesh = Mesh(
        nx=cfg.mesh.nx,
        ny=cfg.mesh.ny,
    )
    return problem, mat_cfg, mesh

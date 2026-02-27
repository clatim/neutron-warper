from dataclasses import dataclass

@dataclass(kw_only=True)
class NeutronProblem():
    xlen: float
    ylen: float
    # Albedo of boundaries
    xmin_alpha: float
    xmax_alpha: float
    ymin_alpha: float
    ymax_alpha: float


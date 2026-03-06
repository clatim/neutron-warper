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

    def __post_init__(self):
        assert(0 <= self.xmin_alpha <= 1)
        assert(0 <= self.xmax_alpha <= 1)
        assert(0 <= self.ymin_alpha <= 1)
        assert(0 <= self.ymax_alpha <= 1)

from dataclasses import dataclass
from nwarper.mixins import Printable


@dataclass
class Settings(Printable):
    convergence_criteria: float
    problem_type: str

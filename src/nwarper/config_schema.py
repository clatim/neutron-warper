from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class ValidRegions(Enum):
    box = 0
    circle = 1


class ValidProblemType(Enum):
    criticality = 0
    fixed = 1


@dataclass
class AxisSpacing:
    spacing: List[List[float]]


@dataclass
class MeshConfig:
    x: AxisSpacing
    y: AxisSpacing


@dataclass
class BoundaryConditions:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class MaterialConfig:
    sigt: float
    D: float | None = 0.0
    sigf: float | None = 0.0
    sigs: float | None = 0.0
    fixed_source: float | None = 0.0
    chi: float | None = 0.0
    nu: float | None = 0.0


@dataclass
class DomainConfig:
    size: List[float]
    boundary_conditions: BoundaryConditions


@dataclass
class RegionConfig:
    shape: ValidRegions
    material: str
    bounds: List[float] | None = None
    centre: List[float] | None = None
    radius: float | None = None


@dataclass
class SettingsConfig:
    problem_type: ValidProblemType
    convergence_criteria: float


@dataclass
class Config:
    settings: SettingsConfig
    domain: DomainConfig
    mesh: MeshConfig
    materials: Dict[str, MaterialConfig]
    regions: List[RegionConfig]

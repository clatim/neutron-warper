from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class ValidRegions(Enum):
    box = 0
    circle = 1


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
    D: float
    sigt: float
    fixed_source: float | None = 0.0


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
class Config:
    domain: DomainConfig
    mesh: MeshConfig
    materials: Dict[str, MaterialConfig]
    regions: List[RegionConfig]

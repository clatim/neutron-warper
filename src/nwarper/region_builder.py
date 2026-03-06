from .shapes import BoxRegion, CircleRegion
from nwarper.material import MaterialConfiguration


class RegionException(Exception):
    pass


def build_regions(cfg, materials: MaterialConfiguration):

    regions = []

    for r in cfg:
        mid = materials.id(r.material)

        if r.shape == "box":
            regions.append(BoxRegion(r.bounds, mid))

        elif r.shape == "circle":
            regions.append(CircleRegion(r.centre, r.radius, mid))

        else:
            # TODO: dont use a system error
            raise RegionException(f"Region {r} not recognised")

    return regions

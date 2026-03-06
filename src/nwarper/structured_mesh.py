import numpy as np
from nwarper.mixins import Printable


def build_axis(blocks):
    """ Given input like

    build_axis([0, 0.5, 2], [0.5, 1.0, 4])

    makes the mesh boundaries for that"""
    coords = []

    for start, end, dx in blocks:
        xs = np.arange(start, end, dx)
        coords.extend(xs)

    # this is adding the last entry?
    coords.append(blocks[-1][1])

    return np.array(coords)

class StructuredMesh(Printable):

    def __init__(self, x_nodes, y_nodes):
        self.x = x_nodes
        self.y = y_nodes

        # Define the centres of cells
        self.xc = 0.5 * (x_nodes[:-1] + x_nodes[1:])
        self.yc = 0.5 * (y_nodes[:-1] + y_nodes[1:])

        self.nx = len(self.xc)
        self.ny = len(self.yc)

        # Make the mesh
        self.Xc,  self.Yc = np.meshgrid(self.xc, self.yc, indexing="ij")

    @classmethod
    def from_config(cls, cfg):
        x_nodes = build_axis(cfg.x.spacing)
        y_nodes = build_axis(cfg.y.spacing)

        return cls(x_nodes, y_nodes)

from nwarper.mixins import Printable


class Region(Printable):

    def __init__(self, material_id):
        self.material_id = material_id

    def mask(self, x, y) -> bool:
        """ Mask to say if x and y is in this region"""
        raise NotImplementedError

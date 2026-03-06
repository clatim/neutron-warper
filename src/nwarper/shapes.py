from nwarper.region import Region

class BoxRegion(Region):

    def __init__(self, bounds, material_id):
        super().__init__(material_id)
        self.x0, self.y0, self.x1, self.y1 = bounds

    def mask(self, x, y):
        return (
            (x >= self.x0)
            & (x <= self.x1)
            & (y >= self.y0)
            & (y <= self.y1)
        )

class CircleRegion(Region):

    def __init__(self, centre, radius, material_id):
        super().__init__(material_id)
        self.cx, self.cy = centre
        self.r = radius

    def mask(self, x, y):
        return (x - self.cx)**2 + (y - self.cy)**2 <= self.r **2

import warp as wp
import numpy as np
from warp import fem


@fem.integrand
def unity(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    """
    See this warp documentation:
    https://nvidia.github.io/warp/domain_modules/fem.html#integrands
    """
    # NOTE: Integrands can't be used directly from `wp.launch`
    # must be called through `fem.integrate` or `fem.interpolate`
    return u(s)


@wp.kernel
def simple_kernel(
    a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), c: wp.array(dtype=float)
):

    tid = wp.tid()

    x = a[tid]
    y = b[tid]

    r = wp.dot(x, y)

    c[tid] = r


def call_simple_kernel(n):
    a = np.ones((n, 3), dtype=np.float32)
    a = wp.from_numpy(a, dtype=wp.vec3)
    b = 2 * np.ones((n, 3), dtype=np.float32)
    b = wp.from_numpy(b, dtype=wp.vec3)

    c = wp.zeros(n, dtype=float)

    wp.launch(kernel=simple_kernel, dim=n, inputs=[a, b], outputs=[c])
    print(c)


@wp.kernel
def make_field(centre: wp.vec3, radius: float, field: wp.array3d(dtype=float)):
    i, j, k = wp.tid()

    p = wp.vec3(float(i), float(j), float(k))

    d = wp.length(p - centre) - radius

    field[i, j, k] = d


def call_make_field(n):

    field = wp.array3d(dtype=float, shape=n)
    centre = wp.vec3(2, 2, 2)
    radius = 1.0

    wp.launch(kernel=make_field, dim=n, inputs=[centre, radius], outputs=[field])
    print(field.numpy())


if __name__ == "__main__":
    wp.init()
    wp.set_device("cpu")
    call_simple_kernel(n=10)
    call_make_field(n=(3, 3, 3))

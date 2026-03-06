""" At the moment this is just a silly test to get something here

Need to think about adding more
"""

def test_axis_generation():
    from nwarper.structured_mesh import build_axis

    blocks = [
        [0, 0.5, 0.1],
        [0.5, 1.1, 0.3],
    ]

    x = build_axis(blocks)

    assert x[0] == 0
    assert x[-1] == 1.1

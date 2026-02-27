# How to run

`uv run nwarper --help` to see command line args 

`uv run nwarper` to run

# Motivation

I want to try and solve the neutron diffusion equation using warp.

There is an example of solving the 2d and 3d diffusion equations [here](https://github.com/NVIDIA/warp/blob/main/warp/examples/fem/example_diffusion.py)
but I will only use this as a reference for now.

To get this done I am going to follow the basic workflow suggested [here](https://nvidia.github.io/warp/domain_modules/fem.html#basic-workflow)
which gives the following steps:

- [x] Define a `Geometry`.
    - See `setup_geometry`
- [x] Define a `FunctionSpace`.
    - See `setup_functionspace`
- [x] Define an integration domain.
    - See `setup_integration_domain`
- [x] Integrate linear forms to get source terms $b$.
    - See `integrate_linear_form`.
- [x] Integrate bilinear forms to build the system of linear equations $A$.
- [x] Solve $Ax = b$

# Todo

- [x] Set up boundary specific boundary conditions.

    This is probably similar to how the material fields have been set up but could make it independent of the material so that it is uniform over a given side.

    Ended up being just using normals

- [x] Try it on a GPU (for reference the GPU is used is a GeForce RTX 4060)
    - [ ] Check integral of solution between `--device=cpu` and `--device=cuda:0`

- [x] Add in input files. Maybe try OmegaConf?

- [ ] Add problem dependent material assignment

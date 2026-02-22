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

- [ ] Set up boundary specific boundary conditions.

    This is probably similar to how the material fields have been set up

- [ ] Try it on a GPU

- [x] Add in input files. Maybe try OmegaConf?

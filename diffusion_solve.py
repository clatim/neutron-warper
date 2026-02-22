"""
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

"""
import warp as wp
import numpy as np
import warp.fem as fem
from warp.sparse import bsr_axpy
from warp.optim.linear import cg
from IPython import embed
from dataclasses import dataclass
import pyvista

@dataclass(kw_only=True)
class NeutronDiffusion():
    nx: int
    ny: int
    D: float # The diffusion coefficient
    sigt: float
    source_strength: float
    xlen: float
    ylen: float
    alpha: float # Albedo of boundaries


@fem.integrand
def integrate_flux(s: fem.Sample, domain: fem.Domain, solution: fem.Field, trial_function: fem.Field):

    return trial_function(s) * solution(s)

@fem.integrand
def source_term(s: fem.Sample, domain: fem.Domain, test_function: fem.Field, source_strength: float):

    return source_strength * test_function(s)

@fem.integrand
def diffusion_bilinear_form(s: fem.Sample, domain: fem.Domain, trial_function: fem.Field, test_function: fem.Field, D: float, sigt: float):

    pos = domain(s)
    return D * wp.dot(
        fem.grad(test_function, s),
        fem.grad(trial_function, s),
    ) + sigt * test_function(s) * trial_function(s)

@fem.integrand
def diffusion_boundary_bilinear_form(s: fem.Sample, domain: fem.Domain, trial_function: fem.Field, test_function: fem.Field, D: float, sigt: float, alpha: float):

    # return 3.0 / 2.0 * sigt * D * test_function(s) * trial_function(s)
    return 1.0 / 2.0 * (alpha - 1.0) / (alpha + 1.0) * test_function(s) * trial_function(s)

def setup_integration_domain(geo):

    domain = fem.Cells(geo)
    return domain

def setup_geometry(problem: NeutronDiffusion):

    geo = fem.Grid2D(res=(problem.nx, problem.ny), bounds_hi=(problem.xlen, problem.ylen))
    return geo

def setup_functionspace(geo, degree=2):

    functionspace = fem.make_polynomial_space(geo, degree=degree)
    return functionspace

def integrate_linear_form(problem: NeutronDiffusion, func_space: fem.FunctionSpace, domain: fem.Domain, linear_form):

    test = fem.make_test(space=func_space, domain=domain)
    rhs = fem.integrate(linear_form, fields={"test_function": test}, values={"source_strength": problem.source_strength})
    return rhs

def integrate_bilinear_form(problem: NeutronDiffusion, geo: fem.Geometry, func_space: fem.FunctionSpace, domain: fem.Domain):

    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=func_space, domain=boundary)
    bd_trial = fem.make_trial(space=func_space, domain=boundary)
    bd_matrix = fem.integrate(
        diffusion_boundary_bilinear_form, 
        fields={
            "test_function": bd_test,
            "trial_function": bd_trial,
        }, 
        values={
            "D": problem.D, 
            "sigt": problem.sigt,
            "alpha": problem.alpha,
        })

    test = fem.make_test(space=func_space, domain=domain)
    trial = fem.make_trial(space=func_space, domain=domain)
    matrix = fem.integrate(diffusion_bilinear_form, domain=domain, fields={"test_function": test, "trial_function": trial}, values={"D": problem.D, "sigt": problem.sigt})

    bsr_axpy(x=bd_matrix, y=matrix, alpha=-1, beta=1)

    return matrix

def solve_ax_b(A, b):
    x = wp.zeros_like(b)
    final_it, resid_norm, abs_tol = cg(A, b, x=x, maxiter=100, tol=1e-6, check_every=1)
    print(f"{final_it = }, {resid_norm = }, {abs_tol = }")
    return x


if __name__ == "__main__":

    wp.init()
    wp.set_device("cpu")
    problem = NeutronDiffusion(
        nx=20,
        ny=20,
        source_strength=1,
        D=1,
        sigt=1.0,
        xlen=1,
        ylen=1,
        alpha=0.0,
    )
    MAGIC_VACCUM = 0.7104 * 1 / problem.sigt # For extrapolated boundary
    geo = setup_geometry(problem)
    func_space = setup_functionspace(geo)
    domain = setup_integration_domain(geo)
    rhs = integrate_linear_form(problem, func_space, domain, linear_form=source_term)
    matrix = integrate_bilinear_form(problem, geo, func_space, domain)
    x = solve_ax_b(A=matrix, b=rhs)
    field = func_space.make_field()

    # Extract cells, nodes and values
    cells, types = field.space.vtk_cells()
    nodes = field.space.node_positions().numpy()
    values = field.dof_values.numpy()
    positions = np.hstack((nodes, values[:, np.newaxis]))

    # Visualise with pyvista
    grid = pyvista.UnstructuredGrid(cells, types, positions)
    # Normalise really badly
    # x = x.numpy()
    # x = x / x[0]
    print(x)
    grid.point_data["scalars"] = x
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid)
    plotter.show()




    


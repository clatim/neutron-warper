import math
import argparse
import warp as wp
import numpy as np
import warp.fem as fem
from warp.sparse import bsr_axpy
from warp.optim.linear import cg
from IPython import embed
from dataclasses import dataclass
import pyvista
import reader

@dataclass(kw_only=True)
class NeutronDiffusion():
    nx: int
    ny: int
    xlen: float
    ylen: float
    alpha: float # Albedo of boundaries
    materials: List[Material]

@dataclass
class CrosssectionFields():
    D: fem.Field
    sigt: fem.Field
    fixed_source: fem.Field

@dataclass(kw_only=True)
class Material():
    sigt: float
    D: float # The diffusion coefficient
    fixed_source: float


@fem.integrand
def integrate_flux(s: fem.Sample, domain: fem.Domain, solution: fem.Field, trial_function: fem.Field):

    return trial_function(s) * solution(s)

@fem.integrand
def source_term(s: fem.Sample, domain: fem.Domain, test_function: fem.Field, S: fem.Field):

    return S(s) * test_function(s)

@fem.integrand
def diffusion_bilinear_form(s: fem.Sample, domain: fem.Domain, trial_function: fem.Field, test_function: fem.Field, D: fem.Field, sigt: fem.Field):

    return D(s) * wp.dot(
        fem.grad(test_function, s),
        fem.grad(trial_function, s),
    ) + sigt(s) * test_function(s) * trial_function(s)

@fem.integrand
def diffusion_boundary_bilinear_form(s: fem.Sample, domain: fem.Domain, trial_function: fem.Field, test_function: fem.Field, alpha: float):

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

def integrate_linear_form(problem: NeutronDiffusion, func_space: fem.FunctionSpace, domain: fem.Domain, linear_form, xsec_data: CrosssectionFields):

    test = fem.make_test(space=func_space, domain=domain)
    rhs = fem.integrate(linear_form, fields={"test_function": test, "S": xsec_data.fixed_source})
    return rhs

def integrate_bilinear_form(problem: NeutronDiffusion, geo: fem.Geometry, func_space: fem.FunctionSpace, domain: fem.Domain, xsec_data: CrosssectionFields):

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
            "alpha": problem.alpha,
        })

    test = fem.make_test(space=func_space, domain=domain)
    trial = fem.make_trial(space=func_space, domain=domain)
    matrix = fem.integrate(diffusion_bilinear_form, domain=domain, fields={"test_function": test, "trial_function": trial, "sigt": xsec_data.sigt, "D": xsec_data.D})

    bsr_axpy(x=bd_matrix, y=matrix, alpha=-1, beta=1)

    return matrix

def solve_ax_b(A, b):
    x = wp.zeros_like(b)
    final_it, resid_norm, abs_tol = cg(A, b, x=x, maxiter=10000, tol=1e-8, check_every=10)
    print(f"{final_it = }, {resid_norm = }, {abs_tol = }")
    return x

def create_material_field(problem: NeutronDiffusion, property: str, material_ids) -> fem.Field:
    """ Creates a fem.Field as is needed by Warp of a given property.

    Uses material mappings from cells to Materials stored in NeutronDiffusion.
    """

    material_values = []
    for id in material_ids:
        material_values.append(
            getattr(problem.materials[id], property)
        )

    material_values = np.array(material_values)
    material_function_space = fem.make_polynomial_space(geo, degree=0, discontinuous=True)
    material_field = material_function_space.make_field()
    material_field.dof_values = wp.from_numpy(material_values, dtype=float)

    return material_field

def define_problem(filename):
    """ Defines the problem.
    """

    problem_def = reader.read_problem(filename)

    mat_list = []
    for name, mat in problem_def["materials"].items():
        mat_list.append(
            Material(
                D=mat["D"],
                sigt=mat["sigt"],
                fixed_source=mat.get("fixed_source", 0.0),
            )
        )


    problem = NeutronDiffusion(
        nx=problem_def.mesh.nx,
        ny=problem_def.mesh.ny,
        xlen=problem_def.mesh.xlen,
        ylen=problem_def.mesh.ylen,
        alpha=1.0,
        materials=mat_list,
    )

    return problem


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot', 
        help="Plots the solution using vtk libraries",
        action="store_true",
    )
    parser.add_argument(
        'input_file',
    )
    parser.add_argument(
            '--device',
            help=(
                "The default device that warp will use. "
                f"The list of available devices is {wp.get_devices()}"
                ),
            default='cpu',
            )
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)
    problem = define_problem(args.input_file)

    geo = setup_geometry(problem)
    ct = geo.cell_count()
    # Set material Ids
    material_id = [1 for _ in range(ct)]
    # for i in range(math.floor(ct/2)):
    #     material_id[i] = 1
    material_ids = np.array(material_id, dtype=int)


    D: fem.Field = create_material_field(problem, "D", material_ids)
    sigt: fem.Field = create_material_field(problem, "sigt", material_ids)
    fixed_source: fem.Field = create_material_field(problem, "fixed_source", material_ids)
    xsec_data = CrosssectionFields(D, sigt, fixed_source)

    func_space = setup_functionspace(geo)
    domain = setup_integration_domain(geo)
    rhs = integrate_linear_form(problem, func_space, domain, linear_form=source_term, xsec_data=xsec_data)
    matrix = integrate_bilinear_form(problem, geo, func_space, domain, xsec_data)
    x = solve_ax_b(A=matrix, b=rhs)
    print("Solution": x)

    if args.plot:
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




    


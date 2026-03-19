import sys
import h5py
from nwarper.structured_mesh import StructuredMesh
import numpy as np
import warp as wp
import warp.fem as fem
from warp.sparse import bsr_axpy
from warp.optim.linear import cg
import nwarper.reader
from nwarper.neutronproblem import NeutronProblem
from nwarper.cross_section_fields import CrossSectionFields
from nwarper.warp_helper import create_fem_field, setup_functionspace
from nwarper.plotting import visualise_solution, visualise_materials


def calculate_norm(x):
    """This is a dummy to calculate the norm of an array

    It is not a good norm right now.
    """

    # TODO: Make this L2
    return np.linalg.norm(x.numpy())


@fem.integrand
def source_term(
    s: fem.Sample, domain: fem.Domain, test_function: fem.Field, S: fem.Field
):

    return S(s) * test_function(s)


@fem.integrand
def fission_source_term(
    s: fem.Sample,
    domain: fem.Domain,
    test_function: fem.Field,
    nu: fem.Field,
    sigf: fem.Field,
    phi: fem.Field,
):

    return nu(s) * sigf(s) * phi(s) * test_function(s)


@fem.integrand
def diffusion_bilinear_form(
    s: fem.Sample,
    domain: fem.Domain,
    trial_function: fem.Field,
    test_function: fem.Field,
    D: fem.Field,
    sigr: fem.Field,
):

    return D(s) * wp.dot(
        fem.grad(test_function, s),
        fem.grad(trial_function, s),
    ) + sigr(s) * test_function(s) * trial_function(s)


@fem.integrand
def diffusion_boundary_bilinear_form(
    s: fem.Sample,
    domain: fem.Domain,
    trial_function: fem.Field,
    test_function: fem.Field,
    xmin_alpha: float,
    xmax_alpha: float,
    ymin_alpha: float,
    ymax_alpha: float,
):

    n = fem.normal(domain, s)

    if n[0] == -1:
        alpha = xmin_alpha
    elif n[0] == 1:
        alpha = xmax_alpha
    elif n[1] == -1:
        alpha = ymin_alpha
    elif n[1] == 1:
        alpha = ymax_alpha

    return (
        1.0 / 2.0 * (alpha - 1.0) / (alpha + 1.0) * test_function(s) * trial_function(s)
    )


def setup_integration_domain(geo):

    domain = fem.Cells(geo)
    return domain


def setup_geometry(problem: NeutronProblem, mesh: StructuredMesh):

    geo = fem.Grid2D(res=(mesh.nx, mesh.ny), bounds_hi=(problem.xlen, problem.ylen))
    return geo


def integrate_bilinear_form(
    problem: NeutronProblem,
    geo: fem.Geometry,
    func_space: fem.FunctionSpace,
    domain: fem.Domain,
    xsec_data: CrossSectionFields,
):

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
            "xmin_alpha": problem.xmin_alpha,
            "xmax_alpha": problem.xmax_alpha,
            "ymin_alpha": problem.ymin_alpha,
            "ymax_alpha": problem.ymax_alpha,
        },
    )

    test = fem.make_test(space=func_space, domain=domain)
    trial = fem.make_trial(space=func_space, domain=domain)
    matrix = fem.integrate(
        diffusion_bilinear_form,
        domain=domain,
        fields={
            "test_function": test,
            "trial_function": trial,
            "sigr": xsec_data.sigr,
            "D": xsec_data.D,
        },
    )

    bsr_axpy(x=bd_matrix, y=matrix, alpha=-1, beta=1)

    return matrix


def solve_ax_b(A, b, x):
    final_it, resid_norm, abs_tol = cg(
        A, b, x=x, maxiter=10000, tol=1e-8, check_every=10
    )
    print(f"{final_it = }, {resid_norm = }, {abs_tol = }")
    return x


def diffusion_solve(
    input_file,
    output_file=None,
    plot_solution=False,
    plot_geometry=False,
    device="cpu",
    overrides=None,
):

    wp.init()
    wp.set_device(device)
    settings, problem, mesh, materials, regions, material_ids = (
        nwarper.reader.read_problem(input_file, overrides)
    )
    if plot_geometry:
        print("Plotting geometry and exiting")
        visualise_materials(material_ids)
        sys.exit()

    geo = setup_geometry(problem, mesh)

    xsec_data = CrossSectionFields(geo, materials, material_ids)

    func_space = setup_functionspace(geo)
    domain = setup_integration_domain(geo)
    test = fem.make_test(space=func_space, domain=domain)
    fixed_source = fem.integrate(
        source_term,
        fields={"test_function": test, "S": xsec_data.fixed_source},
    )
    matrix = integrate_bilinear_form(problem, geo, func_space, domain, xsec_data)
    phi = wp.ones_like(fixed_source)

    criticality_problem = settings.problem_type == "criticality"

    if not criticality_problem:
        phi = solve_ax_b(A=matrix, b=fixed_source, x=phi)
    else:
        fission_source = wp.ones_like(fixed_source)
        change = float("inf")
        outer_it = 0
        convergence_criteria = settings.convergence_criteria
        lamb = 1.0

        converged = False
        while not converged:
            outer_it += 1
            phi_field = create_fem_field(geo, phi)
            prev_source = fission_source
            fission_source = fem.integrate(
                fission_source_term,
                fields={
                    "test_function": test,
                    "sigf": xsec_data.sigf,
                    "nu": xsec_data.nu,
                    "phi": phi_field,
                },
            )

            prev_lamb = lamb
            lamb = sum(fission_source.numpy()) / (sum(prev_source.numpy()) / prev_lamb)

            phi = solve_ax_b(
                A=matrix, b=fixed_source + fission_source / wp.float64(lamb), x=phi
            )
            change = abs(lamb - prev_lamb) / prev_lamb

            print(f"{outer_it = } {change = } {lamb = }")
            if change < convergence_criteria:
                converged = True

    if output_file:
        with h5py.File(output_file, "w") as h:
            h["flux"] = phi
            if criticality_problem:
                h["keff"] = lamb

    if plot_solution:
        visualise_solution(solution=phi, func_space=func_space)

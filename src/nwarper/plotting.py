import warp.fem as fem
import numpy as np
import pyvista
import plotly.express as px


def visualise_materials(material_ids):
    fig = px.imshow(material_ids)
    fig.show()


def visualise_solution(solution, func_space: fem.FunctionSpace):
    field = func_space.make_field()
    # Extract cells, nodes and values
    cells, types = field.space.vtk_cells()
    nodes = field.space.node_positions().numpy()
    values = field.dof_values.numpy()
    positions = np.hstack((nodes, values[:, np.newaxis]))

    # Visualise with pyvista
    grid = pyvista.UnstructuredGrid(cells, types, positions)
    # Normalise really badly
    # solution = solution.numpy()
    # solution = solution / solution[0]
    # print(solution)
    grid.point_data["scalars"] = solution
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid)
    plotter.show()

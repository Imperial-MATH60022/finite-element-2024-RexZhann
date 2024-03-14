import numpy as np
from . import ReferenceTriangle, ReferenceInterval
from .finite_elements import LagrangeElement, lagrange_points
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation


class FunctionSpace(object):

    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element = element
        cell = element.cell
        self.nodes_per_entity = np.array([len(self.element.entity_nodes.get(d, [0])[0]) for d in range(self.mesh.dim+1)])

        # Calculate the total number of nodes in the function space
        self.node_count = sum(self.nodes_per_entity * np.array([mesh.entity_counts[d] for d in range(mesh.dim + 1)]))

        self.cell_nodes = np.zeros((len(mesh.cell_vertices), element.node_count), dtype=int)

        # Populate the global cell-node list
        for c, cell in enumerate(mesh.cell_vertices):
            indices = []
            for delta in range(mesh.dim + 1):  # Loop over entity dimensions
                entity_count = mesh.entity_counts[delta]
                for epsilon in range(len(element.entity_nodes[delta])):  # Loop over entities of dimension delta

                    if delta < mesh.dim:
                        print(f'mesh : {mesh.adjacency(mesh.dim, delta)}')
                        print(f'c : {c}')
                        print(f'ep : {epsilon}')
                        i = mesh.adjacency(mesh.dim, delta)[c, epsilon]
                    else:
                        i = c

                    G = self.global_node_number(delta, i, mesh, element)
                    N_delta = self.nodes_per_entity[delta]

                    for offset in range(N_delta):
                        indices.append(G + offset)
            
            print(f'cell_nodes : {self.cell_nodes}')
            print(f'indices : {indices}')
            self.cell_nodes[c] = indices

    def global_node_number(self, delta, i, mesh, element):
        """Calculate the first global node number for entity (delta, i)."""
        G = sum(mesh.entity_counts[dim] * self.nodes_per_entity[dim] for dim in range(delta))
        G += i * self.nodes_per_entity[delta]
        return G

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.mesh,
                               self.element)


class Function(object):
    def __init__(self, function_space, name=None):
        """A function in a finite element space. The main role of this object
        is to store the basis function coefficients associated with the nodes
        of the underlying function space.

        :param function_space: The :class:`FunctionSpace` in which
            this :class:`Function` lives.
        :param name: An optional label for this :class:`Function`
            which will be used in output and is useful for debugging.
        """

        #: The :class:`FunctionSpace` in which this :class:`Function` lives.
        self.function_space = function_space

        #: The (optional) name of this :class:`Function`
        self.name = name

        #: The basis function coefficient values for this :class:`Function`
        self.values = np.zeros(function_space.node_count)

    def interpolate(self, fn):
        """Interpolate a given Python function onto this finite element
        :class:`Function`.

        :param fn: A function ``fn(X)`` which takes a coordinate
          vector and returns a scalar value.

        """

        fs = self.function_space

        # Create a map from the vertices to the element nodes on the
        # reference cell.
        cg1 = LagrangeElement(fs.element.cell, 1)
        coord_map = cg1.tabulate(fs.element.nodes)
        cg1fs = FunctionSpace(fs.mesh, cg1)

        for c in range(fs.mesh.entity_counts[-1]):
            # Interpolate the coordinates to the cell nodes.
            vertex_coords = fs.mesh.vertex_coords[cg1fs.cell_nodes[c, :], :]
            node_coords = np.dot(coord_map, vertex_coords)

            self.values[fs.cell_nodes[c, :]] = [fn(x) for x in node_coords]

    def plot(self, subdivisions=None):
        """Plot the value of this :class:`Function`. This is quite a low
        performance plotting routine so it will perform poorly on
        larger meshes, but it has the advantage of supporting higher
        order function spaces than many widely available libraries.

        :param subdivisions: The number of points in each direction to
          use in representing each element. The default is
          :math:`2d+1` where :math:`d` is the degree of the
          :class:`FunctionSpace`. Higher values produce prettier plots
          which render more slowly!

        """

        fs = self.function_space

        d = subdivisions or (
            2 * (fs.element.degree + 1) if fs.element.degree > 1 else 2
        )

        if fs.element.cell is ReferenceInterval:
            fig = plt.figure()
            fig.add_subplot(111)
            # Interpolation rule for element values.
            local_coords = lagrange_points(fs.element.cell, d)

        elif fs.element.cell is ReferenceTriangle:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            local_coords, triangles = self._lagrange_triangles(d)

        else:
            raise ValueError("Unknown reference cell: %s" % fs.element.cell)

        function_map = fs.element.tabulate(local_coords)

        # Interpolation rule for coordinates.
        cg1 = LagrangeElement(fs.element.cell, 1)
        coord_map = cg1.tabulate(local_coords)
        cg1fs = FunctionSpace(fs.mesh, cg1)

        for c in range(fs.mesh.entity_counts[-1]):
            vertex_coords = fs.mesh.vertex_coords[cg1fs.cell_nodes[c, :], :]
            x = np.dot(coord_map, vertex_coords)

            local_function_coefs = self.values[fs.cell_nodes[c, :]]
            v = np.dot(function_map, local_function_coefs)

            if fs.element.cell is ReferenceInterval:

                plt.plot(x[:, 0], v, 'k')

            else:
                ax.plot_trisurf(Triangulation(x[:, 0], x[:, 1], triangles),
                                v, linewidth=0)

        plt.show()

    @staticmethod
    def _lagrange_triangles(degree):
        # Triangles linking the Lagrange points.

        return (np.array([[i / degree, j / degree]
                          for j in range(degree + 1)
                          for i in range(degree + 1 - j)]),
                np.array(
                    # Up triangles
                    [np.add(np.sum(range(degree + 2 - j, degree + 2)),
                            (i, i + 1, i + degree + 1 - j))
                     for j in range(degree)
                     for i in range(degree - j)]
                    # Down triangles.
                    + [
                        np.add(
                            np.sum(range(degree + 2 - j, degree + 2)),
                            (i+1, i + degree + 1 - j + 1, i + degree + 1 - j))
                        for j in range(degree - 1)
                        for i in range(degree - 1 - j)
                    ]))

    def integrate(self):
        """Integrate this :class:`Function` over the domain.

        :result: The integral (a scalar)."""

        raise NotImplementedError

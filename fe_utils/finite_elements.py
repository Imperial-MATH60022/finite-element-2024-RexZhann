import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle


np.seterr(invalid='ignore', divide='ignore')


'''def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    r = list(range(degree + 1))
    r[1], r[-1] = r[-1], r[1]
    if cell.dim == 1:
        return np.array([[i / degree] for i in r])
    elif cell.dim == 2:
        vertices = [(0, 0), (degree, 0), (0, degree)]
        remaining_points = [(i, j) for i in range(degree + 1) for j in range(degree + 1) if i + j <= degree and (i, j) not in vertices]
        points = vertices + remaining_points
        return np.array([[i / degree, j / degree] for i, j in points])
'''

'''
def lagrange_points(cell, degree, obtain_entity=False):
    if cell.dim == 1:
        # For a 1D interval, simply generate linearly spaced points between the vertices.
        vertices = np.array(cell.vertices)
        points = np.linspace(vertices[0, 0], vertices[1, 0], degree + 1)
        entity_node = {}
        if obtain_entity:
            entity_node = {0: {}, 1: {}}
            for idx, point in enumerate(points):
                if idx == 0 or idx =
            return points[:, np.newaxis],  # Ensure correct shape (N, 1) for 1D points
        else:
            return points[:, np.newaxis]


    else:
         # Total points are calculated by the formula (n+1)(n+2)/2
        points = []
        ind = 0
        # Add vertex points
        ver_ind = []
        for vertex in cell.vertices:
            points.append(vertex)
            ver_ind.append(ind)
            ind += 1

        # Add edge points
        edg_ind = []
        edges = [(0, 1), (1, 2), (2, 0)]
        for v0, v1 in edges:
            edge_points = [(1 - i / degree) * cell.vertices[v0] + (i / degree)
                           * cell.vertices[v1] for i in range(1, degree)]
            points.extend(edge_points)
            edg_ind.append(ind)
            ind += 1

        # Add interior points
        int_ind = []
        for i in range(1, degree):
            for j in range(1, degree - i):
                x = i / degree
                y = j / degree
                z = 1 - x - y
                interior_point = x * cell.vertices[0] + y * cell.vertices[1] + z * cell.vertices[2]
                points.append(interior_point)
                int_ind.append(ind)
                ind += 1

        entity_node = {}
        if obtain_entity:
            entity_nodes = {0: {}, 1: {}, 2: {}}

            # Vertex entities
            for idx, vertex in enumerate(cell.vertices):
                entity_nodes[0][idx] = [idx]

            # Edge entities
            for idx, edg_index in enumerate(edg_ind):
                entity_node[1][idx] = edg_index

            # Interior entity
            for idx, int_index in enumerate(int_ind):
                entity_node[2][idx] = int_index

            return np.array(points), entity_node

        else:
            return np.array(points)
        '''

def lagrange_points(cell, degree, obtain_entity_points=False):
    if cell.dim not in [1, 2]:
        raise NotImplementedError("This function only supports 1D and 2D cells for now.")
    
    points = []  # Store the generated Lagrange points here
    entity_node = {dim: {} for dim in range(cell.dim + 1)}  # Initialize the entity_node dictionary
    
    if cell.dim == 1:
        # For a 1D interval, generate points linearly spaced between the vertices.
        vertices = np.array(cell.vertices)
        points = np.linspace(vertices[0, 0], vertices[1, 0], degree + 1)
        points = points[:, np.newaxis]  # Ensure correct shape (N, 1) for 1D points
        
        if obtain_entity_points:
            # 0D entities are the endpoints
            entity_node[0] = {0: [0], 1: [degree]}
            # 1D entity is the whole cell, excluding the endpoints
            entity_node[1] = {0: list(range(1, degree))}

    elif cell.dim == 2:
        # Add vertex points for 2D cell (triangle)
        for vertex in cell.vertices:
            points.append(vertex)
            
        edges = [(0, 1), (1, 2), (2, 0)]
        edge_points_indices = []
        for idx, (v0, v1) in enumerate(edges):
            edge_indices = [idx * (degree + 1)]  # Starting index for this edge's points
            edge_points = [(1 - i / degree) * cell.vertices[v0] + (i / degree) * cell.vertices[v1] for i in range(1, degree)]
            for point in edge_points:
                points.append(point)
                edge_indices.append(len(points) - 1)
            edge_indices.append((idx + 1) % 3)  # Ending vertex of this edge
            edge_points_indices.append(edge_indices)
            if obtain_entity_points:
                entity_node[1][idx] = edge_indices  # Store edge indices
            
        # Add interior points for 2D cell
        interior_points_indices = []
        for i in range(1, degree):
            for j in range(1, degree - i):
                x = i / degree
                y = j / degree
                z = 1 - x - y
                interior_point = x * cell.vertices[0] + y * cell.vertices[1] + z * cell.vertices[2]
                points.append(interior_point)
                interior_points_indices.append(len(points) - 1)
        if obtain_entity_points:
            entity_node[2] = {0: interior_points_indices}  # Store interior indices
        
        if obtain_entity_points:
            # Store vertices in the entity_node dictionary for 2D cell
            for idx, vertex in enumerate(cell.vertices):
                entity_node[0][idx] = [idx]
                
    points = np.array(points)
    
    if obtain_entity_points:
        return points, entity_node
    else:
        return points


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """

    if not grad:

        if cell.dim == 1:  # when dimension is 1

            size = degree + 1
            vandermonde = np.zeros((len(points), size))
            points = [x[0] for x in points]
            for i, point in enumerate(points):
                vandermonde[i, 0] = np.array([0])
                for j in range(0, degree + 1):
                    vandermonde[i, j] = point ** j
                pass

        if cell.dim == 2:
            size = (degree + 2) * (degree + 1) // 2
            vandermonde = np.zeros((len(points), size))

            for i, point in enumerate(points):
                count_deg, count_pos = 0, 0
                while count_deg <= degree:
                    for j in range(count_deg + 1):
                        if count_pos < size:
                            vandermonde[i, count_pos] = point[1] ** j * point[0] ** (count_deg - j)
                            count_pos += 1
                            if j == count_deg:
                                count_deg += 1
                        pass
                    

    else:  # cases where we take the gradient

        # 1-dimensional case
        if cell.dim == 1:
            size = degree + 1
            vandermonde = np.zeros((len(points), size, 1))  # build up the shape of the vandermonde matrix
            for i, point in enumerate(points):
                for i, point in enumerate(points):
                    vandermonde[i, 0] = np.array([0])  # fill in the constant terms
                    for j in range(1, degree + 1):
                        vandermonde[i, j, 0] = np.array([(j * point ** (j-1))]) # fill in the rest
                    pass
                
        # 2-dimensional case
        if cell.dim == 2:
            size = (degree + 2) * (degree + 1) // 2
            vandermonde = np.zeros((len(points), size, 2))  # build up the shape of the vandermonde matrix
            for i, point in enumerate(points):
                count_deg = 0  # set up two pointers, one for the current column position, and the other for the current degree.
                count_pos = 0
                while count_deg <= degree:
                    for j in range(count_deg + 1):
                        if count_pos < size:
                            vandermonde[i, count_pos, 1], vandermonde[i, count_pos, 0] = np.array([j * np.real(np.power(point[1], (j - 1), dtype=complex)) * 
                                                                                                   point[0] ** (count_deg - j)]), np.array([(count_deg - j) * 
                                                                                                   point[1] ** j * np.real(np.power(point[0], (count_deg - j - 1), dtype=complex))])
                            count_pos += 1
                            if j == count_deg:
                                count_deg += 1
                        pass

    return np.nan_to_num(vandermonde)


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            point evaluation node locations on the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with
            entity `(d, i)` of dimension `d` and index `i`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.

        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """

        vand = vandermonde_matrix(self.cell, self.degree, points, grad=grad)
        if not grad:
            result = vand @ self.basis_coefs

        else:
            result = np.einsum("ijk, jl -> ilk", vand, self.basis_coefs)

        return result

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        return [fn(node) for node in self.nodes]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """
        nodes, self.entity_nodes = lagrange_points(cell, degree, obtain_entity_points=True)

        

        
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes)

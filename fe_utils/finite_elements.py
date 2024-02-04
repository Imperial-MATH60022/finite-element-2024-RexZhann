import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle



np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
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
                for j in range(count_deg + 1):
                    if cell.dim == 2 and count_pos < size:
                        vandermonde[i, count_pos] = point[1] ** j * point[0] ** (count_deg - j)
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
                            vandermonde[i, count_pos, 1], vandermonde[i, count_pos, 0] = np.array([j * point[1] ** (j - 1) * point[0] ** (count_deg - j)]), np.array([(count_deg - j) * point[1] ** j * point[0] ** (count_deg - j - 1)])
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
        self.entity_nodes = entity_nodes

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

        return fn(self.nodes)

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
        nodes = lagrange_points(cell, degree)

        self.entity_node = cell.topology

        
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes)

import networkx as nx
from networkx.utils import py_random_state


@py_random_state(7)
def my_scale_free_graph(n, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2,
                     delta_out=0, create_using=None, seed=None):
    """Returns a scale-free directed graph.

    Parameters
    ----------
    n : integer
        Number of nodes in graph
    alpha : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the in-degree distribution.
    beta : float
        Probability for adding an edge between two existing nodes.
        One existing node is chosen randomly according the in-degree
        distribution and the other chosen randomly according to the out-degree
        distribution.
    gamma : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the out-degree distribution.
    delta_in : float
        Bias for choosing nodes from in-degree distribution.
    delta_out : float
        Bias for choosing nodes from out-degree distribution.
    create_using : NetworkX graph constructor, optional
        The default is a MultiDiGraph 3-cycle.
        If a graph instance, use it without clearing first.
        If a graph constructor, call it to construct an empty graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Examples
    --------
    Create a scale-free graph on one hundred nodes::

    >>> G = nx.scale_free_graph(100)

    Notes
    -----
    The sum of `alpha`, `beta`, and `gamma` must be 1.

    References
    ----------
    .. [1] B. Bollobás, C. Borgs, J. Chayes, and O. Riordan,
           Directed scale-free graphs,
           Proceedings of the fourteenth annual ACM-SIAM Symposium on
           Discrete Algorithms, 132--139, 2003.
    """

    def _choose_node(G, distribution, delta, psum):
        cumsum = 0.0
        # normalization
        r = seed.random()
        for n, d in distribution:
            cumsum += (d + delta) / psum
            if r < cumsum:
                break
        return n

    if create_using is None or not hasattr(create_using, '_adj'):
        # start with 3-cycle
        G = nx.empty_graph(3, create_using, default=nx.MultiDiGraph)
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    else:
        G = create_using
    if not G.is_directed():
        raise nx.NetworkXError("DiGraph required in create_using")

    if alpha < 0:
        raise ValueError('alpha must be >= 0.')
    if beta < 0:
        raise ValueError('beta must be >= 0.')
    if gamma < 0:
        raise ValueError('gamma must be >= 0.')

    if abs(alpha + beta + gamma - 1.0) >= 1e-9:
        raise ValueError('alpha+beta+gamma must equal 1.')

    if n > 0 and G.number_of_nodes() == 0:
        G.add_node(0)

    number_of_edges = G.number_of_edges()
    while len(G) < n:
        psum_in = number_of_edges + delta_in * len(G)
        psum_out = number_of_edges + delta_out * len(G)
        r = seed.random()
        # random choice in alpha,beta,gamma ranges
        if r < alpha:
            # alpha
            # add new node v
            v = len(G)
            # choose w according to in-degree and delta_in
            w = _choose_node(G, G.in_degree(), delta_in, psum_in)
        elif r < alpha + beta:
            # beta
            # choose v according to out-degree and delta_out
            v = _choose_node(G, G.out_degree(), delta_out, psum_out)
            # choose w according to in-degree and delta_in
            w = _choose_node(G, G.in_degree(), delta_in, psum_in)
        else:
            # gamma
            # choose v according to out-degree and delta_out
            v = _choose_node(G, G.out_degree(), delta_out, psum_out)
            # add new node w
            w = len(G)

        if v == w or G.has_edge(v, w):
            continue

        G.add_edge(v, w)
        number_of_edges += 1
    return G


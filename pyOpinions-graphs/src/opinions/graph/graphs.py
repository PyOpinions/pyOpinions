from __future__ import annotations
import networkx as nx
from networkx import Graph
from typing import List, Dict

from opinions.graph.generator import my_scale_free_graph


class GraphManager:

    _instance: GraphManager = None
    _graphs: Dict[str, Graph]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GraphManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._graphs = dict()
        return cls._instance

    def register_graph(self, name: str, g: Graph) -> int:
        self._graphs[name] = g
        return len(self._graphs)

    @property
    def graphs(self) -> Dict[str, Graph]:
        return self._graphs

    def give_me_graph(self, graph_name: str, model_name: str, num_nodes: int, graph_params: dict, seed) -> Graph:
        if graph_name in ('castors', 'polluces'):
            if model_name == 'DSFG':
                g = nx.OrderedDiGraph()
                g.name = graph_name
                # alpha = float(graph_params['<alpha>'])
                # gamma = float(graph_params['<gamma>'])
                # delta_in = float(graph_params['<deltaIn>'])
                # delta_out = float(graph_params['<deltaOut>'])
                alpha, gamma, delta_in, delta_out = map(
                    lambda x: float(x), graph_params['<topologyParams>'].split(sep='_'))
                my_scale_free_graph(num_nodes, alpha=alpha, beta=(1. - alpha - gamma), gamma=gamma,
                                           delta_in=delta_in, delta_out=delta_out, create_using=g, seed=seed)
            else:
                raise NotImplementedError('Graph model NOT Yet Implemented: ' + model_name)

        elif graph_name == 'intervals':
            # NOTE: Graph is NOT directed
            g = nx.OrderedGraph()
            g.name = graph_name
            for i in range(0, 2 * num_nodes, 2):
                g.add_edge(i, i+1)
        elif graph_name == 'ego':
            g = nx.OrderedDiGraph()
            g.name = graph_name
            # g.graph['default_ego'] = graph_params['ego']  # restore?
            g.add_edges_from([(i, i) for i in range(num_nodes)], weight=graph_params['ego'])
        else:
            raise ValueError('Graph type Not known')

        # Register Graph
        self.register_graph(graph_name, g)
        return g

    @staticmethod
    def translate_graph(source: Graph, mapping: List[int]):
        # new_edges = [(mapping[v], mapping[w], attr) for v, w, attr in source.edges().data()]
        # g = nx.Graph()
        # g.add_edges_from(new_edges)
        # # Add more properties if you want
        # return g

        if len(mapping) == len(source.nodes):
            # nx.relabel_nodes(source, mapping= lambda x: mapping[x], copy=False)
            offset = len(mapping)
            nx.relabel_nodes(source, mapping=lambda x: offset + mapping[x], copy=False)
            nx.relabel_nodes(source, mapping=lambda x: x - offset, copy=False)
        # elif len(mapping) < len(source.nodes):
        #     map_dict = dict(zip(range(len(mapping)), mapping))
        #     nx.relabel_nodes(source, mapping=map_dict, copy=False)
        else:
            raise ValueError("mapping length is %d while graph length is %d" % (len(mapping), len(source.edges)))

    @staticmethod
    def extract_dictionary(g: Graph, name) -> List:
        return [n_id for (n_id, attr) in g.nodes().data() if attr['name'] == name]

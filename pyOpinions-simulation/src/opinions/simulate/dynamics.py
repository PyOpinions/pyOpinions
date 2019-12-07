from typing import List, Tuple, Any, Dict

from networkx import Graph

import opinions.objects.constants as constants
from opinions.objects.helper import *
from opinions.objects.opinion import OpinionManager, PointOpinion
from opinions.objects.reference import ReferenceManager, Reference


class OpinionDynamics:

    epsilon = None

    def init(self, graphs: Dict[str, Graph]):
        raise NotImplementedError

    def calculate_update(self, graphs: Dict[str, Graph]) -> List[Tuple[int, int, Any]]:
        raise NotImplementedError


class IntervalCoherenceDynamics(OpinionDynamics):
    __doc__ = """Phi or Beta"""

    def __init__(self, parameter: float, function):
        self.parameter = parameter
        self.function = function

    def init(self, graphs: Dict[str, Graph]):
        for g in graphs.values():
            if g.name != 'intervals':
                continue
            # self.epsilon = g.graph['epsilon']
            for i, j in g.edges():
                if OpinionManager.opinion_id_from_ref_id(i) != OpinionManager.opinion_id_from_ref_id(j):
                    raise IndexError('opinion of i != that of j (%d in %d, %d in %d)'
                                     % (i, OpinionManager.opinion_id_from_ref_id(i), j,
                                        OpinionManager.opinion_id_from_ref_id(i)))

    def calculate_update(self, graphs: Dict[str, Graph]) -> List[Tuple[int, int, Any]]:
        ret = []
        parameter = self.parameter
        function = self.function
        for g in graphs.values():
            if g.name != 'intervals':
                continue
            for i, j in g.edges():
                # Calculate only once per edge
                if i < j:
                    beta_effect = function(i, j, parameter)
                    ret.append((i, j, beta_effect))
                    ret.append((j, i, beta_effect))
        return ret


class EdgeEdgeInteractionDynamics(OpinionDynamics):
    __doc__ = """ICaP, INCaP, CoNCaP-phi or ConCaP-beta"""

    def __init__(self, parameter: float, function):
        self.parameter = parameter
        self.function = function

    def init(self, graphs: Dict[str, Graph]):
        for g in graphs.values():
            if g.name not in ('castors', 'polluxes'):
                continue
            # self.epsilon = g.graph['epsilon']
            for i, j in g.edges():
                if i == j:
                    raise IndexError('References i and j are the same (%d , %d)' % (i, j,))

    def calculate_update(self, graphs: Dict[str, Graph]) -> List[Tuple[int, int, Any]]:
        ret = []
        function = self.function
        for g in graphs.values():
            if g.name not in ('castors', 'polluxes'):
                continue
            for i, j in g.edges():
                ret.append((i, j, function(i, j)))
        return ret


class IntervalEdgeInteractionDynamics(OpinionDynamics):
    __doc__ = """FCoNCaP"""
    castors_graph = None
    polluxes_graph = None
    intervals_graph = None

    def __init__(self, parameter: float, function):
        self.parameter = parameter
        self.function = function

    def init(self, graphs: Dict[str, Graph]):
        self.castors_graph = graphs['castors']
        self.polluxes_graph = graphs['polluxes']
        self.intervals_graph = graphs['intervals']

    def calculate_update(self, graphs: Dict[str, Graph]) -> List[Tuple[int, int, Any]]:
        ret = []
        reference_manager = ReferenceManager()
        opinion_manager = OpinionManager()
        one_minus_beta = self.parameter

        # TODO I (may) unify the next castors and polluxes loops into one, using names like my_self & my_brother

        # ----------castors --------------------------------
        for i, j in self.castors_graph.edges:
            if isinstance(opinion_manager.opinions[opinion_manager.opinion_id_from_ref_id(i)], PointOpinion):
                point_pi_id = i
                point_ci = point_pi = reference_manager.get_reference(i)
                point_cj = reference_manager.get_reference(j)
                v1v2_dot_product = v1_v1_dot_product = 0.
            else:
                point_ci = reference_manager.get_reference(i)
                point_pi_id = list(self.intervals_graph.neighbors(i))[0]  # TODO simplify
                point_pi = reference_manager.get_reference(point_pi_id)
                point_cj = reference_manager.get_reference(j)
                # point_pj_id = list(self.intervals_graph.neighbors(j))[0]  # no need
                # point_pj = reference_manager.get_reference(point_pj_id)  # no need

                vector_v1 = point_ci.anchors - point_pi.anchors
                # equals | v1 | ^ 2
                v1_v1_dot_product: float = vector_v1 @ vector_v1

                # effect of pair (Ci, Pi) on point Cj
                vector_v2 = point_cj.anchors - point_pi.anchors
                # the dot product is |v1| * |v2| * cos (theta)
                v1v2_dot_product: float = vector_v1 @ vector_v2

            if v1v2_dot_product <= 0:
                # point Cj projection is before or on Pi.
                alpha, one_minus_alpha = 0, 1
                y_c = point_pi
            elif v1v2_dot_product >= v1_v1_dot_product:
                # in the line above, notice that v1v1DotProduct actually equals vectorV1LenSqr |v1| * |v1| * cos(0)

                # point Cj projection is on or after Ci.
                alpha, one_minus_alpha = 1, 0
                y_c = point_ci
            else:
                alpha = v1v2_dot_product / v1_v1_dot_product
                one_minus_alpha = 1 - alpha
                y_c = Reference(-1, coordinates=(point_ci.anchors * alpha + point_pi.anchors * one_minus_alpha))

            denominator = constants.EPSILON + y_c.distance_to(point_cj)

            # update p -> c
            if one_minus_alpha:
                nominator = one_minus_beta * one_minus_alpha
                ret.append((point_pi_id, j, nominator / denominator))

            # update c -> c
            if alpha:
                nominator = one_minus_beta * alpha
                ret.append((i, j, nominator / denominator))

        # ----------polluxes --------------------------------
        for i, j in self.polluxes_graph.edges:
            if isinstance(opinion_manager.opinions[opinion_manager.opinion_id_from_ref_id(i)], PointOpinion):
                point_ci_id = i
                point_ci = point_pi = reference_manager.get_reference(i)
                point_pj = reference_manager.get_reference(j)
                v1v2_dot_product = v1_v1_dot_product = 0.
            else:
                point_pi = reference_manager.get_reference(i)
                point_ci_id = list(self.intervals_graph.neighbors(i))[0]  # TODO simplify
                point_ci = reference_manager.get_reference(point_ci_id)
                point_pj = reference_manager.get_reference(j)
                # point_cj_id = list(self.intervals_graph.neighbors(j))[0]
                # point_cj = reference_manager.get_reference(point_cj_id)

                vector_v1 = point_ci.anchors - point_pi.anchors
                # equals | v1 | ^ 2
                v1_v1_dot_product: float = vector_v1 @ vector_v1

                # effect of pair (Ci, Pi) on point Pj
                vector_v2 = point_pj.anchors - point_pi.anchors
                # the dot product is |v1| * |v2| * cos (theta)
                v1v2_dot_product: float = vector_v1 @ vector_v2

            if v1v2_dot_product <= 0:
                # point Pj projection is before or on Pi.
                alpha, one_minus_alpha = 1, 0
                y_p = point_pi
            elif v1v2_dot_product >= v1_v1_dot_product:
                # in the line above, notice that v1v1DotProduct actually equals vectorV1LenSqr |v1| * |v1| * cos(0)

                # point Pj projection is on or after Ci.
                alpha, one_minus_alpha = 0, 1
                y_p = point_ci
            else:
                one_minus_alpha = v1v2_dot_product / v1_v1_dot_product
                alpha = 1 - one_minus_alpha
                y_p = Reference(-1, coordinates=(point_pi.anchors * alpha + point_ci.anchors * one_minus_alpha))

            denominator = constants.EPSILON + y_p.distance_to(point_pj)

            # update c -> p
            if one_minus_alpha:
                nominator = one_minus_beta * one_minus_alpha
                ret.append((point_ci_id, j, nominator / denominator))

            # update p -> p
            if alpha:
                nominator = one_minus_beta * alpha
                ret.append((i, j, nominator / denominator))

        return ret


class EgoDynamics(OpinionDynamics):

    def __init__(self, other_params: float):
        self.other_params = other_params
        # we assume **all** items to be present in the list.
        self.cache_all_value = list()

    def init(self, graphs: Dict[str, Graph]):
        for g in graphs.values():
            if g.name != 'ego':
                continue
            self.cache_all_value.clear()
            for i, j, ego in g.edges.data('weight'):  #, default=g.graph['default_ego']):
                if i != j:
                    raise IndexError(f'i != j ({i}, {j})')
                self.cache_all_value.append(self.other_params * ego / constants.EPSILON)

    def calculate_update(self, graphs: Dict[str, Graph]) -> List[Tuple[int, int, Any]]:
        # default_ego = self.other_params
        ret = []
        for g in graphs.values():
            if g.name != 'ego':
                continue
            cache_all_value = self.cache_all_value
            for i, j in g.edges:
                ret.append((i, j, cache_all_value[i]))
        return ret


class ComplexDynamics (OpinionDynamics):

    __doc__ = """Factory for phi and beta components of ConCaP and FConCaP, """

    all_dynamics: List[OpinionDynamics] = []
    effect_matrix: np.ndarray = None

    def give_me_dynamics(self, model_name: str, params: dict) -> List[OpinionDynamics]:
        if 'epsilon' in params.keys():
            constants.EPSILON = params['epsilon']
        reference_manager = ReferenceManager()

        def phi_function(_: int, __: int):
            return phi

        def beta_function(i: int, j: int, param: float):
            return param / (constants.EPSILON +
                            reference_manager.get_reference(i).distance_to(reference_manager.get_reference(j)))

        if model_name in ['ICaP', 'INCaP']:
            # The difference is that in ICaP, there is a complete graph that connects all C-C or P-P
            dynamics = [EgoDynamics(1), EdgeEdgeInteractionDynamics(1, beta_function)]

        elif model_name == 'CoNCaP-phi':
            phi: float = params['phi']

            dynamics = [EgoDynamics(1 - phi), IntervalCoherenceDynamics(phi, phi_function),
                        EdgeEdgeInteractionDynamics(1 - phi, beta_function)]

        elif model_name == 'CoNCaP-beta':
            beta: float = params['beta']
            dynamics = [EgoDynamics(1 - beta), IntervalCoherenceDynamics(beta, beta_function),
                        EdgeEdgeInteractionDynamics(1 - beta, beta_function)]

        elif model_name == 'FCoNCaP':
            beta: float = params['beta']
            dynamics = [EgoDynamics(1 - beta), IntervalCoherenceDynamics(beta, beta_function),
                        IntervalEdgeInteractionDynamics(1 - beta, beta_function)]

        else:
            raise RuntimeError(f'Model not known: {model_name}.')

        # Register dynamics
        for dyn in dynamics:
            self.all_dynamics.append(dyn)

        return dynamics

    def init(self, graphs: Dict[str, Graph]):
        for dyn in self.all_dynamics:
            dyn.init(graphs)

    def calculate_update(self, graphs: Dict[str, Graph]) -> List[Tuple[int, int, Any]]:
        collected_dynamics: List[Tuple[int, int, float]] = []
        for dyn in self.all_dynamics:
            ds = dyn.calculate_update(graphs)
            collected_dynamics += ds
        return collected_dynamics

    def aggregate_dynamics(self, collected_dynamics: List[Tuple[int, int, Any]]) -> np.ndarray:
        pass


class JustAggregationComplexDynamics(ComplexDynamics):

    def __init__(self, num_references: int):
        self.effect_matrix = np.zeros((num_references, num_references), dtype=float)

    def aggregate_dynamics(self, collected_dynamics: List[Tuple[int, int, Any]]) -> np.ndarray:
        matrix = self.effect_matrix
        for i, j, value in collected_dynamics:
            # notice the order j, i
            matrix[j, i] = value
        # self.normalize_effect_matrix()
        normalize_matrix(matrix)
        return matrix


if __name__ == '__main__':
    complex_dynamics = JustAggregationComplexDynamics(50)
    complex_dynamics.give_me_dynamics('FCoNCaP', {'epsilon': 0.1, 'beta': 0.20})

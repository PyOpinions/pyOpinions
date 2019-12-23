import itertools
import os
import sys
from random import Random
from typing import Dict, List, Sequence, Union

import numpy as np
from networkx import DiGraph, Graph

from opinions.graph.graphs import GraphManager
from opinions.io.opinionsIO import OpinionsIO
from opinions.objects.helper import randomize_matrix, normalize_matrix
from opinions.objects.opinion import OpinionManager, IntervalOpinion, Opinion
from opinions.objects.reference import ReferenceManager, Reference
from opinions.simulate.docopt import docopt
from opinions.simulate.dynamics import JustAggregationComplexDynamics
from opinions.simulate.simulation import Simulation


def egoboost_opinions(egoistics: Union[Sequence[int], Sequence[Opinion]], beta: float, paranoid_coeff: float):
    opinion_manager = OpinionManager()
    graph_manager = GraphManager()
    ego_graph = graph_manager.graphs['ego']
    castors_graph = graph_manager.graphs['castors']
    polluces_graph = graph_manager.graphs['polluces']
    opinions = opinion_manager.opinions
    for opinion_id in egoistics:
        if isinstance(opinion_id, Opinion):
            opinion = opinion_id
        elif isinstance(opinion_id, int):
            opinion = opinions[opinion_id]
        else:
            raise RuntimeError(f'type unsuitable: {type(opinion_id)}.')

        '''
        int inDegree = pointND.getInDegree();
        pointND.setEgo(k * (2 * inDegree + beta * (1 - (2 * inDegree))) );
        '''
        # TODO Again, notice that next two lines works only as we have intervals and points. If we change the
        #  representation or add new opinion types, we would have to rewrite them.
        c_in_degree = castors_graph.in_degree(opinion.references[0])
        p_in_degree = polluces_graph.in_degree(opinion.references[-1])
        min_in = min(c_in_degree, p_in_degree)
        paranoid_ego = paranoid_coeff * 2.0 * min_in + beta * (1 - (2 * min_in))
        for ref in opinion.references:
            ego_graph.add_edge(ref.absolute_id, ref.absolute_id, **{'weight': paranoid_ego})


def move_to_pole(ref: Reference, target_pole: np.ndarray, nu: float):
    nu = nu / 2
    new_coordinates = (ref.anchors * nu) + (target_pole * (1. - nu))
    # normalize point
    new_coordinates /= sum(new_coordinates)
    ref.match(new_coordinates)


def polarize_opinions(egoistics: Union[Sequence[int], Sequence[Opinion]], poles: Sequence[np.ndarray], nu: float, random: Random):
    opinion_manager = OpinionManager()
    opinions = opinion_manager.opinions
    for opinion_id in egoistics:
        # this should be applicable to any number of poles
        index_of_next_target = random.randint(0, len(poles) - 1)

        if isinstance(opinion_id, Opinion):
            opinion = opinion_id
        elif isinstance(opinion_id, int):
            opinion = opinions[opinion_id]
        else:
            raise RuntimeError(f'type unsuitable: {type(opinion_id)}.')

        for ref in opinion.references:
            move_to_pole(ref, poles[index_of_next_target], nu)


def find_stubborn_ids(g: Graph) -> List[int]:
    return [i for i in g.nodes if g.in_degree(i) == 0]


def prepare_simulation(test_params: Dict = None):
    """
    initialize simulation through given parameters.
    This is one of 2 methods to initialize a simulation.
    """

    doc = """Simulate Opinion dynamics (Documentations [almost] complete)

Usage:
  application.py [options]
  application.py [options] [( --topology <topology> <alpha> <gamma> <deltaIn> <deltaOut>)]
  
Options:
  -s, --seed=SEED           Randomization seed (if omitted, use system pseudorandom generator)
  -d, --dimensions=DIMS     Number of dimensions of opinions                [Default: 3]
  -l, --log=LFILE           Log file (if omitted or -, output to stdout)    [Default: -]
  --inFolder=IFOLDER        I don't know. Just in case                      [Default: ./]
  --outFolder=OFOLDER       Where all output files are written              [Default: ./]
  --numOpinions=tOp         Total number of Opinions                        [Default: 256]
  --egoisticPortion=sP      How much % of the total opinions are egoistic   [Default: 0.1]
  --selectFrom=clsNames     comma separated class names to fill the required 
                            stubborn/egocentric from in order.
  --nu=NU                   Polarization coefficient (1. means half range)  [Default: 0.3]
  --ego=EGO                 Default ego value for all references            [Default: 4.0]
  --beta=BETA               Default interval coherence coefficient value    [Default: 0.1]
  --epsilon=EPSILON         Default interaction distance bias value         [Default: 0.1]
  --egoPortion=EGOISTICS    portion (of 1.0) of opinions who are egocentric [Default: 0.0]
  --manageStubborn          How to manage the stubborn opinions. values are none, 
                            polarizeRef(previously polarizeSingle), and polarizeOpinion
                            (previously polarizeCouple)                     [Default: polarizeOpinion]
  --model=MODEL             The opinion dynamics model                      [Default: FCoNCaP]
  --topology=TOPOLOGY       The interaction graph topology (2 B reorganizd) [Default: DSFG]
  --id=ID                   The simulation ID, including all necessary parameters
  --showGUI                 Show results (Do NOT do it if you are running on a remote server).
  --dt=DT                   Visual step delay in seconds (not yet used)     [Default: 0.0]
  -h, --help                Print the help screen and exit.
  --version                 Prints the version and exits.
  --verbose                 Prints a lot of information details.
"""

    # """
    #  application.py [options] [( --selectFrom <className>... )]
    # --selectFrom point interval interval
    #     numCouples --> numOpinions
    #   --intervalsPortion=iP     How much % of the total opinions are intervals  [Default: 0.9]
    #   -i, --in-folder=IFOLDER   input folder where all means/variances are.     [Default: ./]
    #   -o, --out-folder=OFOLDER  Output folder where all scenarios are written   [Default: ./out]
    # """

    args = docopt(doc, version='3.0.0')

    # Testing purpose
    if test_params is not None:
        args.update(test_params)

    log_arg = args['--log']
    if log_arg == '-':
        log = sys.stdout
    else:
        if not os.path.exists(log_arg):
            dirname = os.path.dirname(log_arg)
            if dirname != '':
                os.makedirs(dirname, exist_ok=True)
        log = open(log_arg, 'w')
    args['log'] = log

    # #################################
    # Preparing simulation starts here
    # #################################

    reference_manager = ReferenceManager()
    opinion_manager = OpinionManager()
    graph_manager = GraphManager()
    random = Random() if args['--seed'] is None else Random(int(args['--seed']))
    args['random'] = random
    args['ego'] = float(args['--ego'])
    args['beta'] = float(args['--beta'])

    total_num_opinions = int(args['--numOpinions'])

    topology = args['--topology']
    # --------------- We are currently working on the opinion ID level -----------------------
    castors_points_graph = graph_manager.give_me_graph('castors', topology, total_num_opinions, args, random)
    polluces_points_graph = graph_manager.give_me_graph('polluces', topology, total_num_opinions, args, random)

    natural_stubborn_ids_in_castors = find_stubborn_ids(castors_points_graph)
    natural_stubborn_ids_in_polluces = find_stubborn_ids(polluces_points_graph)
    min_len = min(len(natural_stubborn_ids_in_castors), len(natural_stubborn_ids_in_polluces))
    natural_stubborn_ids_in_castors = natural_stubborn_ids_in_castors[:min_len]
    natural_stubborn_ids_in_polluces = natural_stubborn_ids_in_polluces[:min_len]
    full_list_of_castors_ids = natural_stubborn_ids_in_castors + [i for i in range(total_num_opinions) if
                                                                  i not in natural_stubborn_ids_in_castors]
    full_list_of_polluces_ids = natural_stubborn_ids_in_polluces + [i for i in range(total_num_opinions) if
                                                                    i not in natural_stubborn_ids_in_polluces]

    opinions_castors_mapping = [item[0] for item in
                                sorted([item for item in enumerate(full_list_of_castors_ids)], key=lambda x: x[1])]
    opinions_polluces_mapping = [item[0] for item in
                                 sorted([item for item in enumerate(full_list_of_polluces_ids)], key=lambda x: x[1])]

    # ================================================================================
    class_names = args['--selectFrom'].split(sep=',')
    num_dimensions = int(args['--dimensions'])
    stubborn_opinions = opinion_manager.give_me_num_opinions(min_len, class_names[0], num_dimensions)
    non_stubborn_opinions = opinion_manager.give_me_num_opinions(total_num_opinions - min_len, class_names[1],
                                                                 num_dimensions)
    all_opinions = stubborn_opinions + non_stubborn_opinions
    # ------------------ Let's work on the reference ID level now --------------------
    # This method (references[0] and references[-1]) in the next 2 lines, is valid only in case we have interval and
    # points. But if we later have other classes or chnage the order of references in the opinion, we must depend
    # on having ref.name in ('castor', 'point') or ref.name in ('pollux', 'point')
    references_castors_mapping = list(map(lambda op_id: opinion_manager.opinions[op_id].references[0].absolute_id,
                                          opinions_castors_mapping))
    references_polluces_mapping = list(map(lambda op_id: opinion_manager.opinions[op_id].references[-1].absolute_id,
                                           opinions_polluces_mapping))
    graph_manager.translate_graph(castors_points_graph, references_castors_mapping)
    graph_manager.translate_graph(polluces_points_graph, references_polluces_mapping)

    # ================================================================================
    # interval_portion = float(args['--intervalsPortion'])  # removed
    # num_interval_opinions = int(total_num_opinions * interval_portion)
    # num_point_opinions = total_num_opinions - num_interval_opinions
    #
    # interval_opinions = opinion_manager.give_me_num_opinions(num_interval_opinions, 'interval', num_dimensions)
    # point_opinions = opinion_manager.give_me_num_opinions(num_point_opinions, 'point', num_dimensions)
    # all_opinions = interval_opinions + point_opinions
    #
    # mapping = [ref.absolute_id for ref in itertools.chain(*[opinion.get_references for opinion in all_opinions])
    #            if ref.name in ('castor', 'point')]
    # graph_manager.translate_graph(castors_points_graph, mapping)
    #
    # mapping = [ref.absolute_id for ref in itertools.chain(*[opinion.get_references for opinion in all_opinions])
    #            if ref.name in ('pollux', 'point')]
    # graph_manager.translate_graph(polluces_points_graph, mapping)

    mapping = [ref.absolute_id for ref in itertools.chain(*[
        opinion.get_references for opinion in all_opinions if isinstance(opinion, IntervalOpinion)])]
    intervals_graph = graph_manager.give_me_graph('intervals', '', len(mapping) // 2, args, random)
    graph_manager.translate_graph(intervals_graph, mapping)

    ego_graph = graph_manager.give_me_graph('ego', '', reference_manager.num_references(), args, None)
    mapping = [ref.absolute_id for ref in ReferenceManager().references]
    graph_manager.translate_graph(ego_graph, mapping)

    # egoboost some opinions
    egoistic_portion = float(args['--egoPortion'])
    num_egoistic_opinions = int(total_num_opinions * egoistic_portion)

    egoboosting_candidates = set()
    stubborn_opinions_2_egoboost = set(stubborn_opinions)
    while len(egoboosting_candidates) < num_egoistic_opinions and stubborn_opinions_2_egoboost:
        egoboosting_candidates.add(stubborn_opinions_2_egoboost.pop())
    non_stubborn_opinions_2_egoboost = set(non_stubborn_opinions)
    while len(egoboosting_candidates) < num_egoistic_opinions and non_stubborn_opinions_2_egoboost:
        egoboosting_candidates.add(non_stubborn_opinions_2_egoboost.pop())
    egoboost_opinions(list(egoboosting_candidates), args['beta'], 5)  # TODO parameterize this "5" later

    opinions_io = OpinionsIO()
    files = opinions_io.create_output_files(args)
    args.update(**files)
    # TODO write graphs and ego values to their respective files here.
    #  Ops: one new challenge: How to indicate different references and their class types?!
    #  I am running out of mind and time and have to stop here.

    num_references = reference_manager.num_references()
    complex_dynamics = JustAggregationComplexDynamics(num_references)
    complex_dynamics.give_me_dynamics(args['--model'], args)
    complex_dynamics.init(graph_manager.graphs)

    # these three lines MUST be after creating all references
    positions_matrix_x = reference_manager.consolidate_positions_matrix_and_share_its_objects()
    randomize_matrix(positions_matrix_x, random)
    normalize_matrix(positions_matrix_x)

    # polarize generic egoistics (both natural stubborn + artificially egoistics)
    nu = float(args['--nu'])  # TODO may be changed for script backwards compatibility
    generic_egoistics = set(stubborn_opinions).union(egoboosting_candidates)
    polarize_opinions(list(generic_egoistics), (np.array([1., 0., 0.]), np.array([0.0, 0.5, 0.5])), nu, random)

    normalize_matrix(positions_matrix_x)

    simulation = Simulation(2000, complex_dynamics, args)
    simulation.set_ready(True)
    simulation.start()


if __name__ == '__main__':
    test_params = {
        # 'ego': 4,
        # 'beta': 0.20,
        # 'epsilon': 0.1,
        '--id': 'test',

        # '<alpha>': 0.0,
        # '<gamma>': 0.129,
        # '<deltaIn>': 10.82,
        # '<deltaOut>': 1.55

        '<alpha>': 0.014,
        '<gamma>': 0.1259,
        '<deltaIn>': 10.0,
        '<deltaOut>': 1.31
    }
    prepare_simulation(test_params)

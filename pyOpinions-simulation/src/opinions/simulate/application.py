import itertools
import sys
import os
from random import Random
from typing import Dict, List, Tuple, Iterable, Sequence

import numpy as np
from networkx.generators import ego

from opinions.graph.graphs import GraphManager
from opinions.objects.helper import randomize_matrix, normalize_matrix
from opinions.objects.opinion import OpinionManager
from opinions.objects.reference import ReferenceManager, Reference
from opinions.simulate.docopt import docopt
from opinions.simulate.dynamics import JustAggregationComplexDynamics
from opinions.simulate.simulation import Simulation


def find_egostics() -> List[int]:  # or List[List[int]]
    # TODO implement
    return [1, 2, 4]
    pass


def egoboost_opinions(egostics: List[int], paranoid_ego_value: float):
    opinion_manager = OpinionManager()
    ego_graph = GraphManager().graphs['ego']
    opinions = opinion_manager.opinions
    for opinion_id in egostics:
        for ref_id in opinions[opinion_id]:
            ego_graph.add_edge(ref_id, ref_id, **{'weight': paranoid_ego_value})


def move_to_pole(ref: Reference, target_pole: np.ndarray, nu: float):
    nu = nu / 2
    new_coordinates = (ref.anchors * nu) + (target_pole * (1. - nu))
    # normalize point
    new_coordinates /= sum(new_coordinates)
    ref.match(new_coordinates)


def polarize_opinions(egostics: List[int], poles: Sequence[np.ndarray], nu: float, random: Random):
    opinion_manager = OpinionManager()
    opinions = opinion_manager.opinions
    for opinion_id in egostics:
        # this should be applicaple to any number of poles
        index_of_next_target = random.randint(0, len(poles) - 1)
        for ref in opinions[opinion_id].references:
            move_to_pole(ref, poles[index_of_next_target], nu)


def prepare_simulation(test_params:Dict = None):
    """
    initialize simulation through given parameters.
    This is one of 2 methods to initialize a simulation.
    """

    doc = """Simulate Opinion dynamics (Documentations [almost] complete)

Usage:
  application.py [options]
  application.py [options] [--topology <topology> <alpha> <gamma> <deltaIn> <deltaOut>]

Options:
  -s, --seed=SEED           Randomization seed (if omitted, use system pseudorandom generator)
  -d, --dimensions=DIMS     Number of dimensions of opinions                [Default: 3]
  -l, --log=LFILE           Log file (if omitted or -, output to stdout)    [Default: -]
  --inFolder=IFOLDER        I don't know. Just in case                      [Default: ./]
  --outFolder=OFOLDER       Where all output files are written              [Default: ./]
  --numOpinions=tOp         Total number of Opinions                        [Default: 256]
  --intervalsPortion=iP     how much % of the total opinions are intervals  [Default: 0.9]
  --nu=NU                   polarization coefficient (1. means half range)  [Default: 0.05]
  --ego=EGO                 Default ego value for all references            [Default: 4.0]
  --beta=BETA               Default interval coherence coefficient value    [Default: 0.1]
  --epsilon=EPSILON         Default interaction distance bias value         [Default: 0.1]
  --egoRatio=EGORATIO       portion (of 1.0) of opinions who are egocentric [Default: 0.0]
  --manageStubborn          how to manage the stupporn opinions. values are none, 
                            polarizeSingle(polarizeRef), and polarizeCouple(polarizeOpinion)
                                                                            [Default: polarizeCouple]
  --model=MODEL             The opinion dynamics model                      [Default: FCoNCaP]
  --topology=TOPOLOGY       THe interaction graph topology                  [Default: DSFG]
  --id=ID                   The simulation ID, including all necessary parameters
  --showGUI                 Show results (Do NOT do it if you are running on a remote server).
  --dt=DT                   Visual step delay in seconds (not yet used)     [Default: 0.0]
  -h, --help                Print the help screen and exit.
  --version                 Prints the version and exits.
"""

# """
#     numCouples --> numOpinions
#   -i, --in-folder=IFOLDER   input folder where all means/variances are.     [Default: ./]
#   -o, --out-folder=OFOLDER  Output folder where all scenarios are written   [Default: ./out]
# """

    args = docopt(doc, version='3.0.0')

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

    if test_params is not None:
        args.update(test_params)
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
    interval_portion = float(args['--intervalsPortion'])
    num_interval_opinions = int(total_num_opinions * interval_portion)
    num_point_opinions = total_num_opinions - num_interval_opinions

    num_dimensions = int(args['--dimensions'])
    interval_opinions = opinion_manager.give_me_num_opinions(num_interval_opinions, 'interval', num_dimensions)
    point_opinions = opinion_manager.give_me_num_opinions(num_point_opinions, 'point', num_dimensions)
    all_opinions = interval_opinions + point_opinions

    intervals_graph = graph_manager.give_me_graph('intervals', '', num_interval_opinions, args, random)
    mapping = [ref.absolute_id for ref in itertools.chain(*[opinion.get_references for opinion in interval_opinions])]
    graph_manager.translate_graph(intervals_graph, mapping)

    topology = args['--topology']
    castors_points_graph = graph_manager.give_me_graph('castors', topology, total_num_opinions, args, random)
    mapping = [ref.absolute_id for ref in itertools.chain(*[opinion.get_references for opinion in all_opinions])
               if ref.name in ('castor', 'point')]
    graph_manager.translate_graph(castors_points_graph, mapping)

    polluxes_points_graph = graph_manager.give_me_graph('polluxes', topology, total_num_opinions, args, random)
    mapping = [ref.absolute_id for ref in itertools.chain(*[opinion.get_references for opinion in all_opinions])
               if ref.name in ('pollux', 'point')]
    graph_manager.translate_graph(polluxes_points_graph, mapping)

    ego_graph = graph_manager.give_me_graph('ego', '', reference_manager.num_references(), args, None)
    mapping = [ref.absolute_id for ref in ReferenceManager().references]
    graph_manager.translate_graph(ego_graph, mapping)

    num_references = reference_manager.num_references()
    complex_dynamics = JustAggregationComplexDynamics(num_references)
    complex_dynamics.give_me_dynamics(args['--model'], args)
    complex_dynamics.init(graph_manager.graphs)

    # these three lines MUST be after creating all references
    positions_matrix_x = reference_manager.consolidate_positions_matrix_and_share_its_objects()
    randomize_matrix(positions_matrix_x, random)
    normalize_matrix(positions_matrix_x)

    egostics = find_egostics()
    # polarize them
    nu = float(args['--nu'])  # TODO may be changed for script backwards compatibility
    polarize_opinions(egostics, (np.array([1., 0., 0.]), np.array([0.0, 0.5, 0.5])), nu, random)

    normalize_matrix(positions_matrix_x)

    simulation = Simulation(2000, complex_dynamics)
    simulation.set_ready(True)
    simulation.start()


if __name__ == '__main__':
    test_params = {
        # 'ego': 4,
        # 'beta': 0.20,
        # 'epsilon': 0.1,

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
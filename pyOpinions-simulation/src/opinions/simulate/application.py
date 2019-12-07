import itertools
import sys
import os
from random import Random
from typing import Dict, List

from opinions.graph.graphs import GraphManager
from opinions.objects.helper import randomize_matrix, normalize_matrix
from opinions.objects.opinion import OpinionManager
from opinions.objects.reference import ReferenceManager
from opinions.simulate.docopt import docopt
from opinions.simulate.dynamics import JustAggregationComplexDynamics
from opinions.simulate.simulation import Simulation


def find_egostics() -> List[int]:  # or List[List[int]]
    # TODO implement
    pass


def egoboost_opinions(egostics: List[int], paranoid_ego_value: float):
    opinion_manager = OpinionManager()
    ego_graph = GraphManager().graphs['ego']
    opinions = opinion_manager.opinions
    for opinion_id in egostics:
        for ref_id in opinions[opinion_id]:
            ego_graph.add_edge(ref_id, ref_id, **{'weight': paranoid_ego_value})

def polarize_opinions(egostics: List[int], random: Random):
    opinion_manager = OpinionManager()
    opinions = opinion_manager.opinions
    # TODO implement
    pass


def prepare_simulation(test_params:Dict = None):
    """
    initialize simulation through given parameters.
    This is one of 2 methods to initialize a simulation.
    """

    doc = """Simulate Opinion dynamics (Documentations not yet full or accurate)

Usage:
  application.py [options]

Options:
  -s, --seed=SEED           Randomization seed (if omitted, use system pseudorandom generator)
  -d, --dimensions=DIMS     Number of dimensions of opinions                [Default: 3]
  -l, --log=LFILE           Log file (if omitted or -, output to stdout)    [Default: -]
  --numOpinions=tOp         Total number of Opinions                        [Default: 256]
  --intervalsPortion=iP     how much % of the total opinions are intervals  [Default: 0.9]
  --show                    Show results (Do NOT do it if you are running on a remote server).
  -h, --help                Print the help screen and exit.
  --version                 Prints the version and exits.
"""

# """
#   -i, --in-folder=IFOLDER   input folder where all means/variances are.     [Default: ./]
#   -o, --out-folder=OFOLDER  Output folder where all scenarios are written   [Default: ./out]
# """

    args = docopt(doc, version='0.1.0')

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

    total_num_opinions = int(args['--numOpinions'])
    interval_portion = float(args['--intervalsPortion'])
    num_interval_opinions = int(total_num_opinions * interval_portion)
    num_point_opinions = total_num_opinions - num_interval_opinions

    interval_opinions = opinion_manager.give_me_num_opinions(num_interval_opinions, 'interval', int(args['--dimensions']))
    point_opinions = opinion_manager.give_me_num_opinions(num_point_opinions, 'point', int(args['--dimensions']))
    all_opinions = interval_opinions + point_opinions

    intervals_graph = graph_manager.give_me_graph('intervals', '', num_interval_opinions, args, random)
    mapping = [ref.absolute_id for ref in itertools.chain(*[opinion.get_references for opinion in interval_opinions])]
    graph_manager.translate_graph(intervals_graph, mapping)

    castors_points_graph = graph_manager.give_me_graph('castors', 'DSFG', total_num_opinions, args, random)
    mapping = [ref.absolute_id for ref in itertools.chain(*[opinion.get_references for opinion in all_opinions])
               if ref.name in ('castor', 'point')]
    graph_manager.translate_graph(castors_points_graph, mapping)

    polluxes_points_graph = graph_manager.give_me_graph('polluxes', 'DSFG', total_num_opinions, args, random)
    mapping = [ref.absolute_id for ref in itertools.chain(*[opinion.get_references for opinion in all_opinions])
               if ref.name in ('pollux', 'point')]
    graph_manager.translate_graph(polluxes_points_graph, mapping)

    ego_graph = graph_manager.give_me_graph('ego', '', reference_manager.num_references(), args, None)
    mapping = [ref.absolute_id for ref in ReferenceManager().references]
    graph_manager.translate_graph(ego_graph, mapping)

    num_references = reference_manager.num_references()
    complex_dynamics = JustAggregationComplexDynamics(num_references)
    complex_dynamics.give_me_dynamics('FCoNCaP', args)
    complex_dynamics.init(graph_manager.graphs)

    # these two lines MUST be after creating all references
    positions_matrix_x = reference_manager.positions_matrix
    randomize_matrix(positions_matrix_x, random)
    # normalize_matrix(positions_matrix_x)

    egostics = find_egostics()
    # TODO polarize them
    polarize_opinions(egostics)

    normalize_matrix(positions_matrix_x)
    # TODO remove one of the normalizations

    simulation = Simulation(2000, complex_dynamics)
    simulation.set_ready(True)
    simulation.start()


if __name__ == '__main__':
    test_params = {
        'ego': 4,
        'beta': 0.20,
        'epsilon': 0.1,
        'alpha': 0.3,
        'gamma': 0.3,
        'deltaIn': 0.2,
        'deltaOut': 0.5
    }
    prepare_simulation(test_params)
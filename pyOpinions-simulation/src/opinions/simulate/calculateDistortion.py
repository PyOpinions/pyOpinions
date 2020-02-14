import os

import math
import sys
from itertools import chain
from pickle import UnpicklingError
from random import Random
from typing import Set, Tuple

import numpy as np
from networkx import DiGraph

from docpie import docpie
from opinions.graph.graphs import GraphManager
from opinions.io.opinionsIO import OpinionsIO
from opinions.objects.opinion import OpinionManager, PointOpinion, IntervalOpinion
from opinions.objects.reference import Reference
from opinions.simulate.votingClasses import *

NUM_EXPERIMENTS = 50
float_and_delimiter = '%8.5E\t'
utilities: List[Tuple[Utility, str]] = [(ExponentialBordaUtility(), 'xb'), (LinearBordaUtility(), 'lb'),
                                        (PluralityUtility(), 'p'), (VetoUtility(), 'v')]
tie_breaking_rule: TieBreakingRule = LexicographicalTieBreakingRule()
voting_rule: VotingRule = PositionalScoringRule()
num_candidates = [3, 6]
candidates_sources = ["100%F", "50%F", "0%F"]


def main(test_params: Dict = None):
    __doc__ = """Calculate distortion according to different candidate/voter permutations per the same scenario.

    Usage:
      calculateDistortion.py [options]
      calculateDistortion.py [options] [( --topology <TOPOLOGY> <topologyParams>)]
      calculateDistortion.py [options] [( --selectFrom <CLSNAMES>)]

    Options:
      -s, --seed=SEED           Randomization seed (if omitted, use system pseudorandom generator)
      -d, --dimensions=DIMS     Number of dimensions of opinions                [Default: 3]
      -l, --log=LFILE           Log file (if omitted or -, output to stdout)    [Default: -]
      --inFolder=IFOLDER        The out folder of the simulation stage          [Default: ./]
      --outFolder=OFOLDER       Where all output files are written              [Default: ./]
      --numOpinions=NOP         Total number of Opinions                        [Default: 256]
      --nu=NU                   Polarization coefficient (1. means half range)  [Default: 0.3]
      --ego=EGO                 Default ego value for all references            [Default: 4.0]
      --beta=BETA               Default interval coherence coefficient value    [Default: 0.1]
      --epsilon=EPSILON         Default interaction distance bias value         [Default: 0.1]
      --egoPortion=EGOISTICS    portion (of 1.0) of opinions who are egocentric [Default: 0.0]
      --manageStubborn=MANGMNT  How to manage the stubborn opinions. values are none, 
                                polarizeRef(previously polarizeSingle), and polarizeOpinion
                                (previously polarizeCouple)                     [Default: polarizeOpinion]
      --model=MODEL             The opinion dynamics model                      [Default: FCoNCaP]
      --topology=TOPOLOGY       The interaction graph topology, followed by,
                                underscore delimited list of parameters thereof [Default: DSFG]
      --selectFrom=CLSNAMES     underscore-separated class names to fill the required
                                stubborn/egocentric from in order.
      --id=ID                   The simulation ID, including all necessary parameters
      --step=STEP               How many steps to take every cycle              [Default: 10]
      --showGUI                 Show results (Do NOT do it if you are running on a remote server).
      --dt=DT                   Visual step delay in seconds (not yet used)     [Default: 0.0]
      -h, --help                Print the help screen and exit.
      --version                 Prints the version and exits.
      --verbose                 Prints a lot of information details.
    """

    # The priority is : hardcoded params for testing >> individual arguments passed to cmd >> params parsed from id
    args = docpie(__doc__, version='3.0.0')

    # # Use ID as a source of parameters as well.
    # id_str: str = args['--id']
    # if id_str:
    #     argv = id2argv(id_str)
    #     args_id = docpie(__doc__, argv=argv, version='3.0.0')
    #
    #     print('passed id')
    #     print(id_str)
    #     print('passed arguments in id')
    #     print(args_id)
    #
    #     # Parameters explicitly coming from ID have a higher priority
    #     args.update(args_id)
    #     print('arguments after update')
    #     print(args)

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

    # print(args)
    random = Random() if args['--seed'] is None else Random(int(args['--seed']))
    args['random'] = random
    args['ego'] = float(args['--ego'])
    opinions_i = OpinionsIO()
    opinions_i.open_input_files(args)
    (graph_manager, reference_manager, opinion_manager) = opinions_i.retrieve_structure_and_topology()

    run_id = str(args['--id'])
    out_folder_arg = args['--outFolder']
    os.makedirs(out_folder_arg, exist_ok=True)
    out_file_str = os.path.join(out_folder_arg, "distortion-%s.csv" % run_id)
    out_file = open(out_file_str, 'w')
    args['out'] = out_file
    summary_file_str = os.path.join(out_folder_arg, "distortionavr-%s.csv" % run_id)
    summary_file = open(summary_file_str, 'w')
    args['summary'] = summary_file

    num_opinions = int(args['--numOpinions'])
    fixed_reference_ids_set = find_fixed_references_ids(graph_manager, args)
    all_opinion_ids = list(range(len(opinion_manager.opinions)))
    all_fixed_opinion_ids = sorted(list({opinion_manager.opinion_id_from_ref_id(i) for i in fixed_reference_ids_set}))
    all_mobile_opinion_ids = sorted(list(set(all_opinion_ids) - set(all_fixed_opinion_ids)))

    list_of__num_experiments__lists_of_candidates_ids: List[List[List[int]]] = []
    for target_num_candidates in num_candidates:
        if target_num_candidates > len(all_fixed_opinion_ids):
            print("%s has only %d fully fixed candidates. Not processed." % (run_id, len(all_fixed_opinion_ids)))
            return
        num_experiments__lists_of_combinations_of_x_candidates_all_from_fixed: List[List[int]] = \
            give_me_x_different_combinations(all_fixed_opinion_ids, target_num_candidates, NUM_EXPERIMENTS, random)
        num_experiments__lists_of_combinations_of_x_candidates_half_from_fixed: List[List[int]] = []
        num_experiments__lists_of_combinations_of_x_candidates_none_from_fixed: List[List[int]] = \
            give_me_x_different_combinations(all_mobile_opinion_ids, target_num_candidates, NUM_EXPERIMENTS, random)
        half_target_num_candidates = int(1.0 * target_num_candidates / 2.0)

        for i in range(NUM_EXPERIMENTS):
            fixed_half = give_me_x_different_combinations(all_fixed_opinion_ids, half_target_num_candidates, 1, random)
            mobile_half = give_me_x_different_combinations(
                all_mobile_opinion_ids, target_num_candidates - half_target_num_candidates, 1, random)
            num_experiments__lists_of_combinations_of_x_candidates_half_from_fixed.append(fixed_half[0] + mobile_half[0])

        list_of__num_experiments__lists_of_candidates_ids.append(
            num_experiments__lists_of_combinations_of_x_candidates_all_from_fixed)
        list_of__num_experiments__lists_of_candidates_ids.append(
            num_experiments__lists_of_combinations_of_x_candidates_half_from_fixed)
        list_of__num_experiments__lists_of_candidates_ids.append(
            num_experiments__lists_of_combinations_of_x_candidates_none_from_fixed)

    all_already_selected_candidates_ids: Set[int] = set()
    list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists: List[Tuple[List[int], List[int]]] = []
    for lst in chain(*list_of__num_experiments__lists_of_candidates_ids):
        all_already_selected_candidates_ids.update(lst)
        list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists.append(
            (lst, sorted(list(set(all_opinion_ids) - set(lst)))))

    all_distances = np.ndarray(shape=(num_opinions, num_opinions), dtype=float)
    to_calculate_dist = np.ndarray((num_opinions, num_opinions), dtype=bool)
    to_calculate_dist.fill(False)
    for i in all_already_selected_candidates_ids:
        for j in range(num_opinions):
            if i == j:
                continue
            to_calculate_dist[i, j] = to_calculate_dist[j, i] = True

    # advance step and skip some steps if needed
    skip = int(args['--step'])
    step_index, prev_step = 0, 0 - skip
    more_to_parse = True
    current_state = None
    max_delta = None
    while more_to_parse:
        try:
            step_index, max_delta, current_state = opinions_i.retrieve_step_delta_and_x()
        except (EOFError, UnpicklingError):
            more_to_parse = False

        if (prev_step + skip > step_index) and more_to_parse:
            continue

        # OK this is the new step
        prev_step = step_index

        # use the new state
        # TODO These coming 2 lines may be of no use after fixing the pickling / unpicking protocol
        for i, ref in enumerate(reference_manager.references):
            ref.set_to(current_state[i])

        # ======== calculate the distances =========================
        for i in all_already_selected_candidates_ids:
            opinion_i = opinion_manager.opinions[i]

            # ================ for caching purpose: Start ==========================
            if isinstance(opinion_i, IntervalOpinion):
                point_ci = opinion_i.references[0]
                point_pi = opinion_i.references[1]
                vector_v1 = point_ci.anchors - point_pi.anchors
                v1_v1_dot_product: float = vector_v1 @ vector_v1
                cache=(point_ci, point_pi, vector_v1, v1_v1_dot_product)
            else:
                cache = None
            # ================ for caching purpose: End ==========================

            for j in all_opinion_ids:
                if not to_calculate_dist[i, j]:  # either i==j or i,j are not connected
                    continue

                opinion_j = opinion_manager.opinions[j]
                point_ci = opinion_i.references[0]
                point_cj = opinion_j.references[0]
                if isinstance(opinion_i, PointOpinion):
                    if isinstance(opinion_j, PointOpinion):
                        distance = point_ci.distance_to(point_cj)
                    else:  # opinion_j is an instance of IntervalOpinion
                        point_pj = opinion_j.references[1]
                        if point_in_segment(point_ci, point_cj, point_pj):
                            distance = 0.
                        else:
                            distance = interval_reference_distance(opinion_j, point_ci)
                else:  # opinion_i is an instance of IntervalOpinion
                    if isinstance(opinion_j, PointOpinion):
                        point_pi = opinion_i.references[1]
                        if point_in_segment(point_cj, point_ci, point_pi):
                            distance = 0.
                        else:
                            distance = interval_reference_distance(opinion_i, point_cj, cache)
                    else:  # Both are intervals
                        point_pi, point_pj = opinion_i.references[1], opinion_j.references[1]
                        if do_intersect(point_ci, point_pi, point_cj, point_pj):
                            distance = 0.
                        else:
                            distance0 = interval_reference_distance(opinion_i, point_cj, cache)
                            if abs(distance0) < 2e-6:
                                distance = 0.
                            else:
                                distance1 = interval_reference_distance(opinion_i, point_pj, cache)
                                distance = distance0 if distance0 < distance1 else distance1
                                if abs(distance) < 2e-6:
                                    distance = 0.

                all_distances[i, j] = all_distances[j, i] = distance  # because the candidate may be a voter some day

        if not step_index:  # if step == 0:
            print_header(utilities, list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists, True, out_file)
            print_header(utilities, list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists, False, summary_file)

        mean, variance = 0.0, 0.0

        out_file.write('%04d\t' % step_index)
        out_file.write(float_and_delimiter % max_delta)
        summary_file.write('%04d\t' % step_index)
        summary_file.write(float_and_delimiter % max_delta)
        for util_function, util_name in utilities:
            for i in range(len(list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists)):
                index_in_group = i % NUM_EXPERIMENTS
                iteration_candidate_ids, iteration_voters_ids = list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists[i]
                winner_id = elect(all_distances, step_index, tie_breaking_rule, voting_rule, util_function,
                                  iteration_candidate_ids, iteration_voters_ids)

                numerator = 0.0
                min_sum_distances = 9000000000  # TODO change
                for candidate_id in iteration_candidate_ids:
                    sum_distances = 0.0
                    for voter_id in iteration_voters_ids:
                        sum_distances += all_distances[candidate_id, voter_id]
                    # Find the numerator (sum of distances per winner)
                    if candidate_id == winner_id:
                        numerator = sum_distances
                    # Find denominator (minimum sum of distances)
                    if sum_distances < min_sum_distances:
                        min_sum_distances = sum_distances

                # distortion = numerator / min_sum_distances
                distortion = (-13.815510557964274 if (numerator <= 1e-6) else math.log(numerator)) - \
                             (-13.815510557964274 if (min_sum_distances <= 1e-6) else math.log(min_sum_distances))
                mean += distortion
                variance += distortion**2
                out_file.write(float_and_delimiter % distortion)
                if index_in_group == NUM_EXPERIMENTS - 1:  # This identifies when we finish the 50 exps of the group
                    #  --- similar check above for the beginning of a group to reset mean and var
                    mean /= NUM_EXPERIMENTS  # divided by 50
                    variance /= NUM_EXPERIMENTS  # divided by 50
                    variance -= mean ** 2
                    variance *= 1.0 * NUM_EXPERIMENTS / (NUM_EXPERIMENTS - 1)  # To produce unbiased variance estimate
                    summary_file.write(float_and_delimiter % mean)
                    summary_file.write(float_and_delimiter % variance)
                    mean = variance = 0.0  # This is the correct reset point ---
                # once we finish the 50 exps of the group we are done

        out_file.write('\n')
        summary_file.write('\n')

    # End of step loop
    print('All done!')
    out_file.close()
    summary_file.close()


def point_in_segment(p: Reference, p1: Reference, q1: Reference):
    """check if point p lies on segment 'p1q1'"""
    o1: int = orientation(p1, q1, p)
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    return o1 == 0 and on_segment(p1, p, q1)


def do_intersect(p1: Reference, q1: Reference, p2: Reference, q2: Reference) -> bool:
    """ test whether line segments 'p1q1' and 'p2q2' intersect.
    Based on code from http://geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    :param p1:
    :type p1:
    :param q1:
    :type q1:
    :param p2:
    :type p2:
    :param q2:
    :type q2:
    :return:
    :rtype:
    """
    # Find the four orientations needed for general and special cases
    o1: int = orientation(p1, q1, p2)
    o2: int = orientation(p1, q1, q2)
    o3: int = orientation(p2, q2, p1)
    o4: int = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False  # Doesn 't fall in any of the above cases


def orientation(p: Reference, q: Reference, r: Reference) -> int:
    """find orientation of ordered triplet (p, q, r).

    See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    for details of below formula.
    :param p: point 1
    :type p: Reference
    :param q: point 2
    :type q: Reference
    :param r: point 3
    :type r: Reference
    :return: 0 if they are colinear, 1 if clockwise, 2 counterclockwise
    :rtype: int
    """
    val = (q.anchors[1] - p.anchors[1]) * (r.anchors[0] - q.anchors[0]) - \
          (q.anchors[0] - p.anchors[0]) * (r.anchors[1] - q.anchors[1])

    if abs(val) < 3e-6:  # middle between 2e-6 and 3e-6
        return 0  # colinear

    return 1 if val > 0 else 2  # clock or counterclock wise


def on_segment(p: Reference, q: Reference, r: Reference) -> bool:
    """check if point q lies on line segment 'pr'"""
    return min(p.anchors[0], r.anchors[0]) <= q.anchors[0] <= max(p.anchors[0], r.anchors[0]) and \
           min(p.anchors[1], r.anchors[1]) <= q.anchors[1] <= max(p.anchors[1], r.anchors[1])


def give_me_x_different_combinations(n_full_list: List[int], r: int, how_many_needed: int, random: Random) \
        -> List[List[int]]:
    n = len(n_full_list)
    ret = []
    if n == r:
        print(f'Warning: N = K = ({n}). We will return ({how_many_needed}) duplicates of the full list')
        for i in range(how_many_needed):
            ret.append(n_full_list)
        return ret
    for i in range(how_many_needed):
        accumulator = set()
        while len(accumulator) < r:
            accumulator.add(random.choice(n_full_list))
        ret.append(sorted(list(accumulator)))
    return ret


# def ncr(n, r):
#     r = min(r, n-r)
#     numer = reduce(op.mul, range(n, n-r, -1), 1)
#     denom = factorial(r)
#     return numer / denom
#

def interval_reference_distance(opinion_i: IntervalOpinion, point_cj: Reference, cache:Tuple = None) -> float:
    if cache:
        point_ci, point_pi, vector_v1, v1_v1_dot_product = cache
    else:
        point_ci = opinion_i.references[0]
        point_pi = opinion_i.references[1]
        # point_cj = opinion_j.references[0]

        vector_v1 = point_ci.anchors - point_pi.anchors
        # equals | v1 | ^ 2
        v1_v1_dot_product: float = vector_v1 @ vector_v1

    # effect of pair (Ci, Pi) on point Cj
    vector_v2 = point_cj.anchors - point_pi.anchors
    # the dot product is |v1| * |v2| * cos (theta)
    v1v2_dot_product: float = vector_v1 @ vector_v2

    if v1v2_dot_product <= 0:
        # point Cj projection is before or on Pi.
        # alpha, one_minus_alpha = 0, 1
        y_c = point_pi
    elif v1v2_dot_product >= v1_v1_dot_product:
        # in the line above, notice that v1v1DotProduct actually equals vectorV1LenSqr |v1| * |v1| * cos(0)

        # point Cj projection is on or after Ci.
        # alpha, one_minus_alpha = 1, 0
        y_c = point_ci
    else:
        alpha = v1v2_dot_product / v1_v1_dot_product
        one_minus_alpha = 1 - alpha
        y_c = Reference(-1, coordinates=(point_ci.anchors * alpha + point_pi.anchors * one_minus_alpha))

    return y_c.distance_to(point_cj)


def print_header(utilities: List[Tuple[Utility, str]],
                 list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists: List[Tuple[List[int], List[int]]],
                 extended: bool, out_file):
    out_file.write('step\tmaxDelta\t')
    # individual_id: List[str] = []
    # group_size, max = None, None
    utility_names = [name for fun, name in utilities]
    if extended:
        group_size = NUM_EXPERIMENTS
        individual_id = [str(i + 1) for i in range(NUM_EXPERIMENTS)]
        header_limit = len(utilities) * len(list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists)
    else:
        group_size = 2
        individual_id = ['M', 'V']
        header_limit = len(utilities) * len(list_of_fifty_pairs_of_candidates_id_lists_and_voter_id_lists) * group_size \
                       // NUM_EXPERIMENTS

    len_candidates_sources = len(candidates_sources)
    len_candidates = (len(num_candidates) * len_candidates_sources)
    for i in range(header_limit):
        individual_index = i % group_size
        whole_batch = i // group_size  # includes all utilities * all candidate sources * num of candidates
        utility_index = whole_batch // len_candidates
        batch_candidates = whole_batch % len_candidates
        num_candidates_id = batch_candidates // len_candidates_sources
        candidates_source_id = batch_candidates % len_candidates_sources  # 3 here: 100%, 50%, 0%
        out_file.write("U%s_C%d_%s_%s\t" % (utility_names[utility_index], num_candidates[num_candidates_id],
                                            candidates_sources[candidates_source_id], individual_id[individual_index]))

    out_file.write('\n')


def elect(all_distances: np.ndarray, step_index: int, tie_breaking_rule: TieBreakingRule, voting_rule: VotingRule,
          utility_function: Utility, iteration_candidate_ids: List[int], iteration_voters_ids: List[int]):
    all_candidates_scores: List[Dict[int, int]] = []
    for voter_id in iteration_voters_ids:
        candidates_distances_per_voter = {candidate_id: all_distances[candidate_id, voter_id]
                                          for candidate_id in iteration_candidate_ids}
        candidates_score_for_this_voter = utility_function.score_candidates(candidates_distances_per_voter)
        all_candidates_scores.append(candidates_score_for_this_voter)
    return voting_rule.find_winner(all_candidates_scores, tie_breaking_rule)


def id2argv(id_str) -> List[str]:
    params = '--' + id_str.replace(',', ' --').replace('=', ' ')
    ret = ['command']
    ret.extend(params.split(' '))
    return ret


def find_fixed_opinions_ids(graph_manager: GraphManager, args: Dict) -> List[int]:
    opinion_manager = OpinionManager()
    ids = {opinion_manager.opinion_id_from_ref_id(i) for i in find_fixed_references_ids(graph_manager, args)}
    return sorted(list(ids))


def find_fixed_references_ids(graph_manager: GraphManager, args: Dict) -> Set[int]:
    default_ego = float(args['--ego'])
    ego_graph = graph_manager.graphs['ego']
    high_ego = set([i for i, j, wt in ego_graph.edges.data('weight') if wt > default_ego])

    castors_graph: DiGraph = graph_manager.graphs['castors']
    stubborn_castors = set([i for i, in_degree in castors_graph.in_degree() if in_degree == 0])

    polluces_graph: DiGraph = graph_manager.graphs['polluces']
    stubborn_polluces = set([i for i, in_degree in polluces_graph.in_degree() if in_degree == 0])

    return high_ego | stubborn_castors | stubborn_polluces

# ====================================================


if __name__ == '__main__':
    main()

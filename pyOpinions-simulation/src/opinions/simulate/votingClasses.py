# from collections.abc import Collection
import operator
from collections import Counter
from functools import reduce
from typing import Dict, List


class Utility(object):
    def score_candidates(self, distances: Dict[int, float]) -> Dict[int, int]:
        raise NotImplementedError()

    def sort_candidates(self, distances) -> List[int]:
        return [c_id for c_id, val in sorted(distances.items(), key=lambda x: x[1])]


class BordaUtility(Utility):

    def score_candidates(self, distances: Dict[int, float]) -> Dict[int, int]:
        sorted_candidates_ids = self.sort_candidates(distances)
        return self.fill_ret_map(sorted_candidates_ids)

    def fill_ret_map(self, sorted_candidates_ids: List[int]) -> Dict[int, int]:
        raise NotImplementedError()


class LinearBordaUtility(BordaUtility):

    def fill_ret_map(self, sorted_candidates_ids: List[int]) -> Dict[int, int]:
        max_score = len(sorted_candidates_ids) - 1
        ret = {c_id: max_score - order for order, c_id in enumerate(sorted_candidates_ids)}
        # return dict(sorted(ret.items()))  # This (unnecessary step) must be cpython 3.6+ or python 3.7+
        return ret


class ExponentialBordaUtility(BordaUtility):
    def fill_ret_map(self, sorted_candidates_ids: List[int]) -> Dict[int, int]:
        ln = len(sorted_candidates_ids)
        ret = {c_id: 1 << (ln - order - 1) for order, c_id in enumerate(sorted_candidates_ids)}
        return ret


class VetoUtility(Utility):
    def score_candidates(self, distances: Dict[int, float]):
        sorted_candidates_ids = self.sort_candidates(distances)
        last_order = len(sorted_candidates_ids) - 1
        ret = {c_id: (0 if order == last_order else 1) for order, c_id in enumerate(sorted_candidates_ids)}
        return ret


class PluralityUtility(Utility):

    def score_candidates(self, distances: Dict[int, float]):
        sorted_candidates_ids = self.sort_candidates(distances)
        ret = {c_id: (1 if order == 0 else 0) for order, c_id in enumerate(sorted_candidates_ids)}
        return ret


class TieBreakingRule:
    def break_tie(self, candidate_ids: List[int]) -> int:
        raise NotImplementedError()


class LexicographicalTieBreakingRule(TieBreakingRule):
    def break_tie(self, candidate_ids: List[int]) -> int:
        return sorted(candidate_ids)[0]


class VotingRule:
    def find_winner(self, all_candidates_scores: List[Dict[int, int]], tie_breaking_rule: TieBreakingRule):
        raise NotImplementedError()


class PositionalScoringRule(VotingRule):
    def find_winner(self, all_candidates_scores: List[Dict[int, int]], tie_breaking_rule: TieBreakingRule):
        # the Counter inside the map() function is the class (constructor ?)
        # found in https://www.geeksforgeeks.org/python-sum-list-of-dictionaries-with-same-key/
        votes_per_candidate_counter: Counter[int, int] = reduce(operator.add, map(Counter, all_candidates_scores))
        # votes_per_candidate_dict: Dict[int, int] = dict(votes_per_candidate_counter)
        top_score = votes_per_candidate_counter.most_common(1)[0][1]
        toppers = [c_id for c_id, val in votes_per_candidate_counter.items() if val == top_score]
        if len(toppers) == 1:
            return toppers[0]
        else:
            return tie_breaking_rule.break_tie(toppers)


from __future__ import annotations

from typing import List

from opinions.objects.reference import Reference, ReferenceManager

ref_id_to_opinion_id: List[int] = []


class Opinion:
    references: List[Reference] = []
    _exact_number_of_references = -1  # unrealistic value
    reference_names: List[str] = None

    def __init__(self, references: List[Reference]):
        self.check_num_references(references)
        # super(IntervalOpinion, self).__init__(references)
        self.references = references
        self.name_references()

    @property
    def get_references(self) -> List[Reference]:
        return self.references
        # TODO make sure it returns the reference list itself, not a shallow copy

    def check_num_references(self, references):
        if len(references) != type(self)._exact_number_of_references:
            raise ValueError(
                f'Number of references passed in ('
                f'{len(references)}) not equal to expected ({type(self)._exact_number_of_references})')

    def name_references(self):
        for i, ref in enumerate(self.references):
            ref.name = type(self).reference_names[i]


class IntervalOpinion(Opinion):
    _exact_number_of_references = 2
    reference_names = ['castor', 'pollux']

    # def __init__(self, references: List[Reference]):
    #     self.check_num_references(references)
    #     super(IntervalOpinion, self).__init__(references)
    #     self.name_references()


class PointOpinion(Opinion):
    _exact_number_of_references = 1
    reference_names = ['point']

    # def __init__(self, references: List[Reference]):
    #     self.check_num_references(references)
    #     super(PointOpinion, self).__init__(references)
    #     self.name_references()


class OpinionManager:

    _instance: OpinionManager = None
    opinions: List[Opinion] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OpinionManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.opinions = []
        return cls._instance

    def give_me_num_opinions(self, num_opinions: int, op_type: str, num_dimensions: int) -> List[Opinion]:
        ret: List[Opinion] = []
        for i in range(num_opinions):
            ret.append(self.give_me_an_opinion(op_type, num_dimensions))
        return ret

    def give_me_an_opinion(self, op_type: str, num_dimensions: int):
        reference_manager = ReferenceManager()
        if op_type == 'interval':
            castor = reference_manager.new_reference([0.0] * num_dimensions)
            pollux = reference_manager.new_reference([0.0] * num_dimensions)
            new_opinion = IntervalOpinion([castor, pollux])
            id = len(self.opinions)
            self.opinions.append(new_opinion)
            ref_id_to_opinion_id.append(id)
            ref_id_to_opinion_id.append(id)
            return new_opinion
        elif op_type == 'point':
            p = reference_manager.new_reference([0.0] * num_dimensions)
            new_opinion = PointOpinion([p])
            id = len(self.opinions)
            self.opinions.append(new_opinion)
            ref_id_to_opinion_id.append(id)
            return new_opinion
        else:
            raise NotImplementedError('Opinion Type not known: '+op_type)

    @classmethod
    def opinion_id_from_ref_id(cls, ref_id: int) -> int:
        # TODO accept iterable
        # TODO Empty iterable -> all references
        return ref_id_to_opinion_id[ref_id]


if __name__ == '__main__':
    op_manager = OpinionManager()
    opinions = op_manager.give_me_num_opinions(5, 'interval', 3)
    print(opinions)
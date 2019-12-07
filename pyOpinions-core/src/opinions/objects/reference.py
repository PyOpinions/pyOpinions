from __future__ import annotations
from typing import Union, List

import numpy as np
from opinions.objects.helper import distance_between

class Reference:
    __doc__ = """
    """
    anchors: np.ndarray = np.array([], dtype='f4')
    _dimensions: int = 0  # instance (not class) variable
    _absolute_id: int = -1
    _name: str = None

    def __init__(self, ref_id=-1, coordinates: Union[List[float], np.ndarray] = None):
        self._absolute_id = ref_id
        self.set_to(np.array(coordinates) if isinstance(coordinates, list) else coordinates)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def num_dimensions(self):
        return self._dimensions

    def move_to(self, new_coordinate: List[float]):
        if len(new_coordinate) != self._dimensions:
            raise ValueError("Number of dimensions is not compatible")
        self.anchors = np.array(new_coordinate, copy=True)

    def match(self, new_coordinates: np.ndarray):
        """
        Set the coordinates to passed-in value, without any checks.
        Use this function with caution
        :param new_coordinates:
        :type new_coordinates:
        :return:
        :rtype:
        """
        self.anchors[:] = new_coordinates

    def add(self, delta: Union[np.ndarray, Reference]) -> Reference:
        d = delta if isinstance(delta, np.ndarray) else delta.anchors
        if np.shape(d) != np.shape(self.anchors):
            raise Exception("Not compatible")
        self.anchors += d
        return self

    def set_to(self, anchors: np.array):
        """
        Just set_to. Do not check or do anything
        :param anchors:
        :type anchors:
        :return:
        :rtype:
        """
        self.anchors = anchors
        self._dimensions = len(anchors)

    @property
    def absolute_id(self) -> int:
        return self._absolute_id

    def distance_to(self, other: Reference):
        return distance_between(self.anchors, other.anchors)


class ReferenceManager:
    _instance: ReferenceManager = None
    _references: List[Reference] = None
    _positions_matrix: np.ndarray = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ReferenceManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._references = []
        return cls._instance

    def new_reference(self, coordinates: Union[list, np.ndarray]) -> Reference:
        next_id = len(self._references)
        ret = Reference(next_id, coordinates=coordinates)
        self.register_reference(ret)
        return ret

    def register_reference(self, ref: Reference) -> int:
        self._references.append(ref)
        return len(self._references)

    def get_reference(self, id: int) -> Reference:
        return self._references[id]

    @property
    def references(self) -> List[Reference]:
        return self._references

    def num_references(self):
        return len(self._references)

    def consolidate_positions_matrix_and_share_its_objects(self) -> np.ndarray:
        """
        Create the Singleton matrix and share its objects with references.
        Notice that once the matrix is created, it can not be modified (only positions can be updated).
        Call it ONLY AFTER  you have created all the references.
        """
        if self._positions_matrix is None:
            arr = [ref.anchors for ref in self._references]
            self._positions_matrix = np.array(arr, copy=False)
            # now the other way: share objects between the ndarray and anchors
            for i, ref in enumerate(self._references):
                ref.anchors = self._positions_matrix[i]
        return self._positions_matrix



if __name__ == '__main__':
    rm = ReferenceManager()
    print(rm)
    r1 = rm.new_reference([1, 2, 3])
    print(r1, r1.anchors)
    rm = ReferenceManager()
    print(rm)
    r2 = rm.new_reference([10, 11, 12])
    print(r2, r2.anchors)
    r1.add(r2)
    print(r1, r1.anchors)

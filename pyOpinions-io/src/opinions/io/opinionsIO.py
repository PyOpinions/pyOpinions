import os
from pickle import Pickler, Unpickler
from typing import Dict

from typing.io import BinaryIO

from opinions.interfaces.interfaces import SimulationListener


class OpinionsIO (SimulationListener):
    structurePickler: Pickler = None
    structureUnpickler: Unpickler = None
    topologyPickler: Pickler = None
    topologyUnpickler: Unpickler = None
    xPickler: Pickler = None
    xUnpickler: Unpickler = None

    files: Dict[str, BinaryIO]

    def __init__(self):
        pass

    def open_output_files(self, args: Dict, protocol=4) -> Dict:
        out_folder_arg = args['--outFolder']
        if not os.path.exists(out_folder_arg):
            os.makedirs(out_folder_arg, exist_ok=True)

        run_id = args['--id']
        topology_file_path = os.path.join(out_folder_arg, 'topology-%s.log' % (run_id,))
        topology_file = open(topology_file_path, 'wb')
        structure_file_path = os.path.join(out_folder_arg, 'structure-%s.log' % (run_id,))
        structure_file = open(structure_file_path, 'wb')
        x_file_path = os.path.join(out_folder_arg, 'x-%s.log' % (run_id,))
        x_file = open(x_file_path, 'wb')
        # d_file  # Do you really want it?

        ret = dict()
        ret['topologyFile'] = topology_file
        ret['structureFile'] = structure_file
        ret['xFile'] = x_file
        # ret['dFile'] = d_file  # Really need it?
        self.files = ret

        self.structurePickler = Pickler(structure_file, protocol=protocol)
        self.topologyPickler = Pickler(topology_file, protocol=protocol)
        self.xPickler = Pickler(x_file, protocol=protocol)

        return ret

    def simulation_starting(self, state):
        self.topologyPickler.dump(state[0])
        self.structurePickler.dump(state[1])

    def simulation_started(self, state):
        self.xPickler.dump(state)

    def update(self, state):
        # later add the ability to store change in topology graph
        if len(state) == 2:
            self.xPickler.dump(state)
        elif len(state) == 2:
            self.xPickler.dump((state[0], state[1]))
            self.topologyPickler.dump(state[2])
        else:
            raise RuntimeError('Unknown state length / structure : ' + str(state))

    def simulation_ending(self, state):
        self.xPickler.dump(state)

    def simulation_ended(self, state):
        self.files['xFile'].close()
        self.files['topologyFile'].close()
        self.files['structureFile'].close()


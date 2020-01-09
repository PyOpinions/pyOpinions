import os
from pickle import Pickler, Unpickler
from typing import Dict, Tuple
import numpy as np

from typing.io import BinaryIO

from opinions.graph.graphs import GraphManager
from opinions.interfaces.interfaces import SimulationListener
from opinions.objects.opinion import OpinionManager
from opinions.objects.reference import ReferenceManager


class OpinionsIO (SimulationListener):
    structurePickler: Pickler = None
    structureUnpickler: Unpickler = None
    topologyPickler: Pickler = None
    topologyUnpickler: Unpickler = None
    xPickler: Pickler = None
    xUnpickler: Unpickler = None

    outfiles: Dict[str, BinaryIO]

    def __init__(self):
        pass

    def open_input_files(self, args: Dict) -> Dict:
        try:
            in_folder_arg = args['--inFolder']
            if not os.path.exists(in_folder_arg):
                raise FileNotFoundError(f'Folder not found {in_folder_arg}')

            run_id = str(args['--id'])
            # run_folder = os.path.join(in_folder_arg, run_id)
            # if not os.path.exists(run_folder):
            #     raise FileNotFoundError(f'Folder not found {run_folder}')

            topology_file_path = os.path.join(in_folder_arg, 'topology-%s.log' % (run_id,))
            # Ugly solution that works only on windows
            # topology_file_path = '\\\\?\\'+topology_file_path
            topology_file = open(topology_file_path, 'rb')
            structure_file_path = os.path.join(in_folder_arg, 'structure-%s.log' % (run_id,))
            structure_file = open(structure_file_path, 'rb')
            x_file_path = os.path.join(in_folder_arg, 'x-%s.log' % (run_id,))
            x_file = open(x_file_path, 'rb')
            # d_file  # Do you really want it?
        except FileNotFoundError as err:
            raise RuntimeError(
                # f"Error: {err}\n"
                f"If you are using windows. This may be caused by a problem related to long path names.\n"
                f"To fix it on windows 10 1607 and later, The registry key "
                f"HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem LongPathsEnabled "
                f"(Type: REG_DWORD) must exist and be set to 1.\n"
                f"Refer to https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file?#enable-long-paths-in-windows-10-version-1607-and-later for details."
                )

        ret = dict()
        ret['topologyFile'] = topology_file
        ret['structureFile'] = structure_file
        ret['xFile'] = x_file
        # ret['dFile'] = d_file  # Really need it?

        self.structureUnpickler = Unpickler(structure_file)
        self.topologyUnpickler = Unpickler(topology_file)
        self.xUnpickler = Unpickler(x_file)

        return ret

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
        self.outfiles = ret

        self.structurePickler = Pickler(structure_file, protocol=protocol)
        self.topologyPickler = Pickler(topology_file, protocol=protocol)
        self.xPickler = Pickler(x_file, protocol=protocol)

        return ret

    def simulation_starting(self, state):
        """Save graph_manager, reference_manager, and opinion_manager"""
        self.topologyPickler.dump(state[0])
        self.structurePickler.dump(state[1])
        self.structurePickler.dump(state[2])

    def retrieve_structure_and_topology(self) -> Tuple[GraphManager, ReferenceManager, OpinionManager]:
        """retrieve graph_manager, reference_manager, and opinion_manager"""
        return self.topologyUnpickler.load(), self.structureUnpickler.load(), self.structureUnpickler.load()

    def simulation_started(self, state):
        self.xPickler.dump(state)

    def retrieve_step_and_x(self) -> Tuple[int, np.ndarray]:
        return self.xUnpickler.load()

    def update(self, state):
        # later add the ability to store change in topology graph
        if len(state) == 2:
            self.xPickler.dump(state)
        elif len(state) > 2:
            self.xPickler.dump((state[0], state[1]))
            self.topologyPickler.dump(state[2])
        else:
            raise RuntimeError('Unknown state length / structure : ' + str(state))

    def simulation_ending(self, state):
        self.xPickler.dump(state)

    def simulation_ended(self, state):
        self.outfiles['xFile'].close()
        self.outfiles['topologyFile'].close()
        self.outfiles['structureFile'].close()

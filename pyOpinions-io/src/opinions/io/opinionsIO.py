import os
from typing import Dict


class OpinionsIO:

    def create_output_files(self, args: Dict) -> Dict:
        out_folder_arg = args['--outFolder']
        if not os.path.exists(out_folder_arg):
            os.makedirs(out_folder_arg, exist_ok=True)

        run_id = args['--id']
        topology_file_path = os.path.join(out_folder_arg, 'topology-%s.log' % (run_id,))
        topology_file = open(topology_file_path, 'w')
        structure_file_path = os.path.join(out_folder_arg, 'structure-%s.log' % (run_id,))
        structure_file = open(structure_file_path, 'w')
        x_file_path = os.path.join(out_folder_arg, 'x-%s.log' % (run_id,))
        x_file = open(x_file_path, 'w')
        # d_file  # Do you really want it?

        ret = dict()
        ret['topologyFile'] = topology_file
        ret['structureFile'] = structure_file
        ret['xFile'] = x_file
        # ret['dFile'] = d_file  # Really need it?

        return ret

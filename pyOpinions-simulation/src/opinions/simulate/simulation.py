from __future__ import annotations

from threading import Thread
from typing import Dict, List

from opinions.graph.graphs import GraphManager
from opinions.interfaces.interfaces import SimulationListener
from opinions.objects.constants import *
from opinions.objects.helper import *
from opinions.objects.reference import ReferenceManager
from opinions.simulate.dynamics import ComplexDynamics


class Simulation(Thread):
    current_step: int = 0
    end_step: int = -1
    ready = False
    verbose = False
    complex_dynamics_d: ComplexDynamics = None

    listeners: List[SimulationListener] = []

    def __init__(self, end_step: int, complex_dynamics: ComplexDynamics, args: Dict):
        super().__init__()
        # self.graphManager = GraphManager()
        self.end_step = end_step
        self.complex_dynamics_d = complex_dynamics
        self.args = args
        self.verbose = args['--verbose']

    def load_simulation(self, path: str):
        """
        initialize simulation from a restart file.
        This is other one of 2 methods to initialize a simulation.
        """
        self.ready = True
        # TODO complete
        raise NotImplementedError()

    def set_ready(self, ready:bool):
        self.ready = ready

    def add_listener(self, listener: SimulationListener):
        self.listeners.append(listener)

    def remove_listener(self, listener: SimulationListener):
        self.listeners.remove(listener)

    def run(self) -> None:
        if not self.ready:
            raise RuntimeError('Simulation NOT ready yet!. call set_ready(True) or load_simulation() first')

        # xFile = self.args['xFile']
        verbose = self.verbose

        end_step = self.end_step
        step = self.current_step
        forever: bool = self.end_step < 0
        converged: bool = False
        x = ReferenceManager().consolidate_positions_matrix_and_share_its_objects()
        normalize_matrix(x)

        graphs = GraphManager().graphs
        complex_dynamics_d = self.complex_dynamics_d
        updates = complex_dynamics_d.calculate_update(graphs)
        d = complex_dynamics_d.aggregate_dynamics(updates)
        # normalize_matrix(d)  # Already called inside aggregate_dynamics(updates)

        if verbose:
            self.print_x_and_d(step, x, d)  # i.e. to the stdout
        for listener in self.listeners:
            listener.simulation_started((step, x))

        while forever or step < end_step:
            # now I have transformation (effects) matrix and x (opinions) matrix

            # update opinions
            # opinionsMatrix = opinionsMatrix x effectMatrix
            temp_x = d @ x

            # calculate the total system update (total absolute distance)
            total_abs_dist = max_distance_between(x, temp_x)
            converged = total_abs_dist < DEFAULT_CONVERGENCE_PRECISION  # (oneOverNSquare / step)

            normalize_matrix(temp_x)
            x[:, :] = temp_x[:, :]

            updates = complex_dynamics_d.calculate_update(graphs)
            d = complex_dynamics_d.aggregate_dynamics(updates)
            # normalize_matrix(d)  # Already called inside aggregate_dynamics(updates)

            # ==============simulation step proper ends here ================
            print('Step = %d, Total Diff = %8.5E, Converged = %r' % (step, total_abs_dist, converged))

            if verbose:
                # self.print_x_and_d(step, x, d)
                self.print_x_and_d(step, temp_x, d)  # TODO Using temp_x is a work around, not a final solution
            # self.print_x_and_d(step, x, d, file=xFile)
            for listener in self.listeners:
                # listener.update((step, x))
                listener.update((step, temp_x))  # TODO Using temp_x instead of x is a work around, not a final solution

            if converged:
                break

            step += 1
        self.current_step = step
        for listener in self.listeners:
            listener.simulation_ended((step, x))

    def print_x_and_d(self, step: int, x: np.ndarray, d: np.ndarray, file=None):
        # TODO Move function to somewhere else (another helper class)
        print(f'Step = {step}', file=file)
        print(np.array2string(x.transpose(), max_line_width=9999999999, threshold=30000000, precision=6,
                              floatmode='fixed', prefix=''), file=file)
        print(file=file)
        # TODO print D (if you really want)

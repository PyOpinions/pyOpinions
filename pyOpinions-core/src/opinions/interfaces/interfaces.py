
class SimulationListener:

    def simulation_starting(self, state):
        raise NotImplementedError()

    def simulation_started(self, state):
        raise NotImplementedError()

    def update(self, state):
        raise NotImplementedError()

    def simulation_ending(self, state):
        raise NotImplementedError()

    def simulation_ended(self, state):
        raise NotImplementedError()

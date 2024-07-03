from numba import njit
import numpy as np

from stochastic_matching.simulator.simulator import Simulator


def make_qs_core(edge_selector):
    def core_simulator(arrivals, graph, n_steps, queue_size,
                       trafic, queue_log, steps_done):
        n, max_queue = queue_log.shape

        for age in range(n_steps):
            for j in range(n):
                queue_log[j, queue_size[j]] += 1

            # Draw an arrival
            node = arrivals.draw()
            queue_size[node] += 1
            if queue_size[node] == max_queue:
                return steps_done + age + 1

            best_edge = edge_selector(graph=graph, queue_size=queue_size, node=node)

            if best_edge > -1:
                trafic[best_edge] += 1
                queue_size[graph.nodes(best_edge)] -= 1
        return steps_done + age + 1

    return njit(core_simulator)  # (core_simulator)


class QueueSize(Simulator):

    def set_state(self):
        super().set_state()
        self.state['queue_size'] = np.zeros(self.model.n, dtype=int)

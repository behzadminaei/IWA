from bwm_solver import BWM_Solver_SciPy
import numpy as np

class IntervalBWM_SciPy:
    """
    Multi-decision-maker Interval BWM
    Input:
        BO[k][j][i] = BO vector for DM k, criterion j, alternative i
        OW[k][j][i] = OW vector for DM k, criterion j, alternative i
    """

    def __init__(self, BO, OW, K, C, A):
        self.BO = BO
        self.OW = OW
        self.K  = K
        self.C  = C
        self.A  = A
        self.IP = [None]*K

    def run(self):
        for k in range(self.K):
            interval_matrix = np.zeros((self.C, self.A, 2))

            for j in range(self.C):
                solver = BWM_Solver_SciPy(
                    self.BO[k][j],
                    self.OW[k][j]
                )

                result = solver.solve_all()

                for i in range(self.A):
                    interval_matrix[j, i, 0] = result["lower"][i]
                    interval_matrix[j, i, 1] = result["upper"][i]

            self.IP[k] = interval_matrix

        return self.IP

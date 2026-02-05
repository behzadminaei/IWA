import numpy as np
from scipy.optimize import minimize

class BWM_Solver_SciPy:
    """
    Solves BWM for a single decision maker & single criterion.
    Computes:
        - Exact weights
        - Lower bounds
        - Upper bounds
    Uses SciPy SLSQP (nonlinear, nonconvex).
    """

    # Add this validation method to BWM_Solver_SciPy class

    def _validate_and_detect_indices(self):
        """
        Auto-detect Best and Worst indices and validate BWM data structure.
        Raises informative errors if data is incorrect.
        """
        # Find all indices where value is 1
        bo_ones = np.where(np.isclose(self.BO, 1.0, atol=1e-6))[0]
        ow_ones = np.where(np.isclose(self.OW, 1.0, atol=1e-6))[0]
        
        # Validation: Must have exactly one "1" in each vector
        if len(bo_ones) == 0:
            raise ValueError(f"BO vector must contain exactly one value of 1 (best criterion). Got: {self.BO}")
        if len(bo_ones) > 1:
            raise ValueError(f"BO vector contains multiple 1's at indices {bo_ones}. Only one best criterion allowed.")
        
        if len(ow_ones) == 0:
            raise ValueError(f"OW vector must contain exactly one value of 1 (worst criterion). Got: {self.OW}")
        if len(ow_ones) > 1:
            raise ValueError(f"OW vector contains multiple 1's at indices {ow_ones}. Only one worst criterion allowed.")
        
        best_idx = int(bo_ones[0])
        worst_idx = int(ow_ones[0])
        
        # Additional validation: All BO values should be >= 1, all OW values should be >= 1
        if np.any(self.BO < 1 - 1e-6):
            raise ValueError(f"All BO values must be >= 1. Got minimum: {np.min(self.BO)}")
        if np.any(self.OW < 1 - 1e-6):
            raise ValueError(f"All OW values must be >= 1. Got minimum: {np.min(self.OW)}")
        
        return best_idx, worst_idx

    def __init__(self, best_to_others, others_to_worst,
             best_index=None, worst_index=None):
    
        self.BO = np.array(best_to_others, float)
        self.OW = np.array(others_to_worst, float)
        self.n  = len(self.BO)

        # Auto-detect with validation if not given
        if best_index is None or worst_index is None:
            auto_best, auto_worst = self._validate_and_detect_indices()
            if best_index is None:
                best_index = auto_best
            if worst_index is None:
                worst_index = auto_worst
        else:
            # If manually provided, still validate
            if not np.isclose(self.BO[best_index], 1.0, atol=1e-6):
                raise ValueError(f"BO[{best_index}] should be 1 (best criterion), got {self.BO[best_index]}")
            if not np.isclose(self.OW[worst_index], 1.0, atol=1e-6):
                raise ValueError(f"OW[{worst_index}] should be 1 (worst criterion), got {self.OW[worst_index]}")

        self.B = best_index
        self.W = worst_index
    
        self.weights = None
        self.e_star  = None
        self.lower   = None
        self.upper   = None

    # -------------------------------------------------------
    # Model (1): minimize e
    # -------------------------------------------------------
    def solve_exact(self):

        def objective(x):
            return x[-1]      # minimize e

        def ineq(x):
            w = x[:-1]
            e = x[-1]
            cons = []

            for i in range(self.n):
                cons.append(e - abs(w[self.B]/w[i] - self.BO[i]))
                cons.append(e - abs(w[i]/w[self.W] - self.OW[i]))

            return np.array(cons)

        def eq(x):
            w = x[:-1]
            return np.sum(w) - 1

        # initial guess
        x0 = np.ones(self.n+1)
        x0[:-1] /= self.n
        x0[-1] = 1

        bounds = [(1e-6, None)]*self.n + [(0, None)]

        res = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type':'ineq', 'fun': ineq},
                {'type':'eq',   'fun': eq},
            ],
            options={'maxiter':1000, 'ftol':1e-9}
        )

        w = np.clip(res.x[:-1], 0, None)
        w = w / w.sum()

        self.weights = w
        self.e_star  = res.x[-1]

        return w, self.e_star

    # -------------------------------------------------------
    # Model (2): Lower bounds
    # -------------------------------------------------------
    def solve_lower_bounds(self):
        assert self.e_star is not None, "Run solve_exact() first."

        L = np.zeros(self.n)

        for t in range(self.n):

            def objective(p):
                return p[t]

            def ineq(p):
                cons = []
                for i in range(self.n):
                    cons.append(self.e_star - abs(p[self.B]/p[i] - self.BO[i]))
                    cons.append(self.e_star - abs(p[i]/p[self.W] - self.OW[i]))
                return np.array(cons)

            def eq(p):
                return np.sum(p) - 1

            p0 = np.ones(self.n)/self.n
            bounds = [(1e-6, None)]*self.n

            res = minimize(
                fun=objective,
                x0=p0,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type':'ineq','fun':ineq},
                    {'type':'eq',  'fun':eq}
                ],
                options={'maxiter':600, 'ftol':1e-9}
            )

            # Validate optimization result
            if not res.success:
                print(f"Warning: Lower bound optimization failed for index {t}")
                print(f"  Message: {res.message}")
                # Use fallback: exact weight - small margin, floored at 0.0
                fallback = max(0.0, self.weights[t] - 0.1) if self.weights is not None else 0.0
                L[t] = fallback
            else:
                value = res.x[t]
                # Validate the result is in reasonable range [0, 1]
                if value < 0:
                    print(f"Warning: Lower bound {value:.6f} is negative for index {t}, clipping to 0")
                    L[t] = 0.0
                elif value > 1.0:
                    print(f"Warning: Lower bound {value:.6f} exceeds 1.0 for index {t}, clipping to 1.0")
                    L[t] = 1.0
                else:
                    L[t] = value

        # Final validation: ensure lower bounds are consistent with exact weights
        if self.weights is not None:
            for t in range(self.n):
                if L[t] > self.weights[t]:
                    print(f"Warning: Lower bound {L[t]:.6f} > exact weight {self.weights[t]:.6f} for index {t}, adjusting")
                    L[t] = min(L[t], self.weights[t])  # Ensure lower <= exact

        self.lower = L
        return L

    # -------------------------------------------------------
    # Model (3): Upper bounds
    # -------------------------------------------------------
    def solve_upper_bounds(self):
        assert self.e_star is not None, "Run solve_exact() first."

        U = np.zeros(self.n)

        for t in range(self.n):

            def objective(p):
                return -p[t]     # maximize

            def ineq(p):
                cons = []
                for i in range(self.n):
                    cons.append(self.e_star - abs(p[self.B]/p[i] - self.BO[i]))
                    cons.append(self.e_star - abs(p[i]/p[self.W] - self.OW[i]))
                return np.array(cons)

            def eq(p):
                return np.sum(p) - 1

            p0 = np.ones(self.n)/self.n
            bounds = [(1e-6, None)]*self.n

            res = minimize(
                fun=objective,
                x0=p0,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type':'ineq','fun':ineq},
                    {'type':'eq',  'fun':eq}
                ],
                options={'maxiter':600, 'ftol':1e-9}
            )

            # Validate optimization result
            if not res.success:
                print(f"Warning: Upper bound optimization failed for index {t}")
                print(f"  Message: {res.message}")
                # Use fallback: exact weight + small margin, capped at 1.0
                fallback = min(1.0, self.weights[t] + 0.1) if self.weights is not None else 1.0
                U[t] = fallback
            else:
                value = -res.fun  # Convert from minimization to maximization result
                # Validate the result is in reasonable range [0, 1]
                if value < 0:
                    print(f"Warning: Upper bound {value:.6f} is negative for index {t}, clipping to 0")
                    U[t] = 0.0
                elif value > 1.0:
                    print(f"Warning: Upper bound {value:.6f} exceeds 1.0 for index {t}, clipping to 1.0")
                    U[t] = 1.0
                else:
                    U[t] = value

        # Final validation: ensure upper bounds are consistent with lower bounds and exact weights
        if self.lower is not None:
            for t in range(self.n):
                if U[t] < self.lower[t]:
                    print(f"Warning: Upper bound {U[t]:.6f} < lower bound {self.lower[t]:.6f} for index {t}, adjusting")
                    U[t] = self.lower[t] + 1e-6  # Ensure upper >= lower
        
        if self.weights is not None:
            for t in range(self.n):
                if U[t] < self.weights[t]:
                    print(f"Warning: Upper bound {U[t]:.6f} < exact weight {self.weights[t]:.6f} for index {t}, adjusting")
                    U[t] = max(U[t], self.weights[t])  # Ensure upper >= exact

        self.upper = U
        return U

    # -------------------------------------------------------
    def solve_all(self):
        self.solve_exact()
        self.solve_lower_bounds()
        self.solve_upper_bounds()

        return {
            "weights": self.weights,
            "e_star": self.e_star,
            "lower": self.lower,
            "upper": self.upper
        }

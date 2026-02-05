import numpy as np
from scipy.optimize import minimize

class BiLevelOptimizer:
    """
    Bi-level optimization for criteria weighting based on hyper-entropy.
    
    Minimizes the maximum weighted hyper-entropy deviation to assign
    higher weights to criteria with higher agreement (lower He).
    """
    
    def __init__(self, cloud_matrix, A, C):
        """
        Parameters:
        -----------
        cloud_matrix : ndarray of shape (A, C, 3)
            Cloud decision matrix where [:, :, 2] contains He values
        A : int
            Number of alternatives
        C : int
            Number of criteria
        """
        self.cloud_matrix = cloud_matrix
        self.A = A
        self.C = C
        
        # Extract hyper-entropy matrix (A × C)
        self.He = cloud_matrix[:, :, 2]
        
        self.weights = None
        self.optimal_value = None
    
    def solve(self):
        """
        Solve the bi-level optimization:
        
        minimize max_{i,j} |He_ij * W_j - He_{i_0,j_0} * W_{j_0}|
        
        subject to:
            Σ W_j = 1
            W_j ≥ 0 for all j
        
        Returns:
        --------
        weights : ndarray of shape (C,)
            Optimal criteria weights
        """
        
        def objective(x):
            """
            x = [W_1, W_2, ..., W_C, t]
            where t is the auxiliary variable for min-max reformulation
            """
            W = x[:-1]  # Criteria weights
            t = x[-1]   # Max deviation variable
            return t
        
        def inequality_constraints(x):
            """
            Reformulate max problem as constraints:
            |He_ij * W_j - He_{i_0,j_0} * W_{j_0}| ≤ t for all i,j,i_0,j_0
            
            Which becomes:
            He_ij * W_j - He_{i_0,j_0} * W_{j_0} ≤ t
            -He_ij * W_j + He_{i_0,j_0} * W_{j_0} ≤ t
            
            For scipy: constraint ≥ 0, so we return: t - |difference|
            """
            W = x[:-1]
            t = x[-1]
            
            constraints = []
            
            # For all pairs (i,j) and (i_0, j_0)
            for i in range(self.A):
                for j in range(self.C):
                    for i_0 in range(self.A):
                        for j_0 in range(self.C):
                            diff = self.He[i, j] * W[j] - self.He[i_0, j_0] * W[j_0]
                            # t - diff ≥ 0
                            constraints.append(t - diff)
                            # t + diff ≥ 0
                            constraints.append(t + diff)
            
            return np.array(constraints)
        
        def equality_constraint(x):
            """Sum of weights = 1"""
            W = x[:-1]
            return np.sum(W) - 1
        
        # Initial guess
        x0 = np.ones(self.C + 1)
        x0[:-1] = 1.0 / self.C  # Equal weights initially
        x0[-1] = 0.1  # Initial t
        
        # Bounds: W_j ≥ 0, t ≥ 0
        bounds = [(0, None)] * self.C + [(0, None)]
        
        # Solve
        result = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': inequality_constraints},
                {'type': 'eq', 'fun': equality_constraint}
            ],
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        # Extract results
        self.weights = result.x[:-1]
        self.weights = np.clip(self.weights, 0, None)
        self.weights = self.weights / self.weights.sum()  # Normalize
        
        self.optimal_value = result.x[-1]
        
        return self.weights
    
    def get_weights(self):
        """Get the computed criteria weights."""
        if self.weights is None:
            raise ValueError("Weights not computed yet. Call solve() first.")
        return self.weights
    
    def display(self):
        """Display the optimal criteria weights."""
        if self.weights is None:
            raise ValueError("Weights not computed yet. Call solve() first.")
        
        print(f"\n{'='*50}")
        print("Optimal Criteria Weights (Bi-Level Optimization)")
        print(f"{'='*50}")
        print(f"Optimal objective value: {self.optimal_value:.6f}\n")
        
        for j in range(self.C):
            print(f"C{j+1}: {self.weights[j]:.6f}")
        
        print(f"\n{'='*50}")
        print("Hyper-Entropy Statistics:")
        print(f"{'='*50}")
        for j in range(self.C):
            mean_he = np.mean(self.He[:, j])
            print(f"C{j+1}: Mean He = {mean_he:.6f}, Weight = {self.weights[j]:.6f}")


# Usage example
if __name__ == "__main__":
    from data_loader import load_bwm_data_from_excel
    from interval_bwm import IntervalBWM_SciPy
    from cloud_dm import CloudDecisionMatrix
    
    # Load and solve
    BO, OW, K, C, A = load_bwm_data_from_excel("data.xlsx")
    IBWM = IntervalBWM_SciPy(BO, OW, K, C, A)
    IP = IBWM.run()
    
    # Construct cloud matrix
    cdm = CloudDecisionMatrix(IP, K, C, A)
    cloud_matrix = cdm.construct()
    
    # Optimize criteria weights
    optimizer = BiLevelOptimizer(cloud_matrix, A, C)
    weights = optimizer.solve()
    optimizer.display()
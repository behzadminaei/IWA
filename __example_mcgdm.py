import numpy as np
from bwm_solver import BWM_Solver_SciPy
from interval_bwm import IntervalBWM_SciPy

# Data from the table
K = 3  # decision makers
C = 3  # criteria
A = 4  # alternatives

# Initialize BO and OW arrays: BO[k][j][i] and OW[k][j][i]
BO = [
    # Decision Maker 1
    [
        [2, 3, 7, 1],  # C1
        [3, 1, 2, 8],  # C2
        [2, 1, 3, 9],  # C3
    ],
    # Decision Maker 2
    [
        [9, 7, 3, 1],  # C1
        [4, 1, 8, 5],  # C2
        [1, 8, 7, 7],  # C3
    ],
    # Decision Maker 3
    [
        [1, 3, 4, 9],  # C1
        [7, 1, 2, 4],  # C2
        [4, 1, 5, 8],  # C3
    ],
]

OW = [
    # Decision Maker 1
    [
        [2, 3, 1, 2],  # C1
        [1, 5, 2, 3],  # C2
        [3, 2, 7, 1],  # C3
    ],
    # Decision Maker 2
    [
        [1, 3, 7, 9],  # C1
        [5, 9, 1, 4],  # C2
        [8, 1, 2, 2],  # C3
    ],
    # Decision Maker 3
    [
        [9, 5, 4, 1],  # C1
        [1, 7, 4, 6],  # C2
        [5, 8, 5, 1],  # C3
    ],
]

# Solve Interval BWM
IBWM = IntervalBWM_SciPy(BO, OW, K, C, A)
IP = IBWM.run()

# Display results
for k in range(K):
    print(f"\n{'='*60}")
    print(f"Decision Maker {k+1} - Interval Weight Matrix")
    print(f"{'='*60}")
    for j in range(C):
        print(f"\nCriterion C{j+1}:")
        for i in range(A):
            lower = IP[k][j, i, 0]
            upper = IP[k][j, i, 1]
            print(f"  A{i+1}: [{lower:.4f}, {upper:.4f}]")
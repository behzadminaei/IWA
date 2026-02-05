from bwm_solver import BWM_Solver_SciPy

# Define your comparison vectors
best_to_others = [2, 3, 7, 1]    # Best criterion compared to all
others_to_worst = [2, 3, 1, 2]   # All criteria compared to worst

# Create solver and get results
solver = BWM_Solver_SciPy(best_to_others, others_to_worst, best_index=3, worst_index=2)
results = solver.solve_all()

# Access results
print("Weights:", results["weights"])
print("Consistency:", results["e_star"])
print("Lower bounds:", results["lower"])
print("Upper bounds:", results["upper"])
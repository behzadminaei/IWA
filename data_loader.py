import pandas as pd
import numpy as np

def load_bwm_data_from_excel(filepath):
    """
    Load BWM data from Excel file with the structure:
    - Row 1: Criteria headers (C1, C2, C3, ...)
    - Row 2: Alternative labels (A1, A2, ...) repeated for each criterion
    - Subsequent rows: Decision maker ID | BO/OW | values for each alternative per criterion
    
    Returns:
        BO: list[list[list]] - BO[k][j][i] for decision maker k, criterion j, alternative i
        OW: list[list[list]] - OW[k][j][i] for decision maker k, criterion j, alternative i
        K: int - number of decision makers
        C: int - number of criteria
        A: int - number of alternatives
    """
    # Read Excel file, skip first row (criteria headers)
    df = pd.read_excel(filepath, header=None)
    
    # Detect structure from row 2 (alternative labels)
    alt_row = df.iloc[1, 2:].dropna()  # Skip first 2 columns (DM ID, BO/OW)
    
    # Count criteria (each criterion has A columns)
    # Find where pattern repeats (A1, A2, A3, A4, A1, A2, ...)
    first_alt = alt_row.iloc[0]
    A = 1
    while A < len(alt_row) and alt_row.iloc[A] != first_alt:
        A += 1
    
    C = len(alt_row) // A  # Number of criteria
    
    # Count decision makers (pairs of BO/OW rows)
    K = (len(df) - 2) // 2  # Skip header rows, then count pairs
    
    # Initialize arrays
    BO = []
    OW = []
    
    # Parse data for each decision maker
    for k in range(K):
        bo_row_idx = 2 + k * 2      # Row index for BO
        ow_row_idx = 2 + k * 2 + 1  # Row index for OW
        
        bo_criteria = []
        ow_criteria = []
        
        # Parse each criterion
        for j in range(C):
            start_col = 2 + j * A  # Skip first 2 columns, then offset by criterion
            end_col = start_col + A
            
            # Extract BO and OW values for this criterion
            bo_values = df.iloc[bo_row_idx, start_col:end_col].values.tolist()
            ow_values = df.iloc[ow_row_idx, start_col:end_col].values.tolist()
            
            bo_criteria.append(bo_values)
            ow_criteria.append(ow_values)
        
        BO.append(bo_criteria)
        OW.append(ow_criteria)
    
    return BO, OW, K, C, A


# Usage from its own (for debugging purposes):
if __name__ == "__main__":
    from bwm_solver import BWM_Solver_SciPy
    from interval_bwm import IntervalBWM_SciPy
    
    # Load data from Excel
    BO, OW, K, C, A = load_bwm_data_from_excel("data.xlsx")
    
    print(f"Loaded: {K} decision makers, {C} criteria, {A} alternatives")
    
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
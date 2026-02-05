"""
Main Pipeline for BWM-Based Multi-Criteria Group Decision Making

This script implements the complete pipeline:
1. Load data from Excel
2. Solve BWM for each decision maker and criterion
3. Construct interval weight matrices
4. Build cloud decision matrix
5. Calculate criteria weights using bi-level optimization
6. Prioritize alternatives
7. Visualize results
"""

import numpy as np
import pandas as pd
from data_loader import load_bwm_data_from_excel
from interval_bwm import IntervalBWM_SciPy
from cloud_dm import CloudDecisionMatrix
from weightcal_bilevel_ooptimization import BiLevelOptimizer
from prioritization import Prioritization
from visualizer import BWMVisualizer


def main():
    """
    Main pipeline function that executes all steps of the BWM-MCGDM process.
    """
    print("="*80)
    print("BWM-Based Multi-Criteria Group Decision Making Pipeline")
    print("="*80)
    
    # Initialize variables for export
    exact_weights = None
    IP = None
    cloud_matrix = None
    prioritizer = None
    
    # ========================================================================
    # Step 1: Load Data from Excel
    # ========================================================================
    print("\n" + "="*80)
    print("Step 1: Loading Data from Excel")
    print("="*80)
    
    excel_file = "Data_G5.xlsx"
    print(f"Loading data from: {excel_file}")
    
    try:
        BO, OW, K, C, A = load_bwm_data_from_excel(excel_file)
        print(f"✓ Successfully loaded data:")
        print(f"  - Decision Makers (K): {K}")
        print(f"  - Criteria (C): {C}")
        print(f"  - Alternatives (A): {A}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # ========================================================================
    # Step 2 & 3: Solve BWM and Construct Interval Weight Matrices
    # ========================================================================
    print("\n" + "="*80)
    print("Step 2-3: Solving BWM and Constructing Interval Weight Matrices")
    print("="*80)
    print("Solving BWM for each decision maker and criterion...")
    
    try:
        IBWM = IntervalBWM_SciPy(BO, OW, K, C, A)
        IP = IBWM.run()
        print(f"✓ Successfully computed interval weight matrices for all {K} decision makers")
        
        # Extract exact weights for display
        from bwm_solver import BWM_Solver_SciPy
        exact_weights = {}
        for k in range(K):
            exact_weights[k] = {}
            for j in range(C):
                solver = BWM_Solver_SciPy(BO[k][j], OW[k][j])
                result = solver.solve_all()
                exact_weights[k][j] = result["weights"]
        
        # Display weights (lower, main/exact, upper)
        print("\n" + "-"*80)
        print("Interval Weights (Lower, Main/Exact, Upper) for Each Decision Maker")
        print("-"*80)
        for k in range(K):
            print(f"\nDecision Maker {k+1}:")
            print("="*80)
            for j in range(C):
                print(f"\n  Criterion C{j+1}:")
                print(f"    {'Alternative':<15} {'Lower':<15} {'Main/Exact':<15} {'Upper':<15}")
                print(f"    {'-'*60}")
                for i in range(A):
                    lower = IP[k][j, i, 0]
                    upper = IP[k][j, i, 1]
                    exact = exact_weights[k][j][i]
                    print(f"    A{i+1:<14} {lower:<15.6f} {exact:<15.6f} {upper:<15.6f}")
    except Exception as e:
        print(f"✗ Error solving BWM: {e}")
        return
    
    # ========================================================================
    # Step 4: Construct Cloud Decision Matrix
    # ========================================================================
    print("\n" + "="*80)
    print("Step 4: Constructing Cloud Decision Matrix")
    print("="*80)
    print("Converting interval weights to cloud models (Ex, En, He)...")
    
    try:
        cdm = CloudDecisionMatrix(IP, K, C, A)
        cloud_matrix = cdm.construct()
        print(f"✓ Successfully constructed cloud decision matrix")
        print(f"  - Shape: {cloud_matrix.shape} (Alternatives × Criteria × 3)")
        
        # Display cloud matrix summary
        print("\nCloud Matrix Summary:")
        print("-" * 80)
        for i in range(A):
            print(f"\nAlternative A{i+1}:")
            for j in range(C):
                Ex = cloud_matrix[i, j, 0]
                En = cloud_matrix[i, j, 1]
                He = cloud_matrix[i, j, 2]
                print(f"  C{j+1}: Ex={Ex:.6f}, En={En:.6f}, He={He:.6f}")
    except Exception as e:
        print(f"✗ Error constructing cloud matrix: {e}")
        return
    
    # ========================================================================
    # Step 5: Calculate Criteria Weights (Bi-Level Optimization)
    # ========================================================================
    print("\n" + "="*80)
    print("Step 5: Calculating Criteria Weights (Bi-Level Optimization)")
    print("="*80)
    print("Optimizing criteria weights based on hyper-entropy...")
    
    try:
        optimizer = BiLevelOptimizer(cloud_matrix, A, C)
        criteria_weights = optimizer.solve()
        print(f"✓ Successfully computed optimal criteria weights")
        optimizer.display()
    except Exception as e:
        print(f"✗ Error in bi-level optimization: {e}")
        return
    
    # ========================================================================
    # Step 6: Prioritize Alternatives
    # ========================================================================
    print("\n" + "="*80)
    print("Step 6: Prioritizing Alternatives")
    print("="*80)
    print("Calculating ranking scores and determining final order...")
    
    try:
        prioritizer = Prioritization(cloud_matrix, criteria_weights, A, C)
        ranking_scores, ranking_order = prioritizer.run()
        print(f"✓ Successfully computed ranking scores")
        prioritizer.display_results()
    except Exception as e:
        print(f"✗ Error in prioritization: {e}")
        prioritizer = None
        return
    
    # ========================================================================
    # Step 7: Visualize Results
    # ========================================================================
    print("\n" + "="*80)
    print("Step 7: Generating Visualizations")
    print("="*80)
    
    try:
        viz = BWMVisualizer(IP=IP, cloud_matrix=cloud_matrix, K=K, C=C, A=A)
        viz.plot_all(save_prefix="bwm_results")
        print(f"✓ Visualizations saved successfully")
        print(f"  - bwm_results_interval_weights.png")
        print(f"  - bwm_results_gmf.png")
        print(f"  - bwm_results_cloud_droplets.png")
    except Exception as e:
        print(f"✗ Error generating visualizations: {e}")
        return
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("Pipeline Execution Complete!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Input file: {excel_file}")
    print(f"  - Decision Makers: {K}")
    print(f"  - Criteria: {C}")
    print(f"  - Alternatives: {A}")
    print(f"\nFinal Ranking (Best to Worst):")
    for rank, alt_idx in enumerate(ranking_order, 1):
        print(f"  Rank {rank}: A{alt_idx+1} (RS = {ranking_scores[alt_idx]:.6f})")
    print("\n" + "="*80)
    
    # ========================================================================
    # Step 8: Export Results to Excel
    # ========================================================================
    print("\n" + "="*80)
    print("Step 8: Exporting Results to Excel")
    print("="*80)
    
    try:
        if exact_weights is not None and IP is not None and cloud_matrix is not None and prioritizer is not None:
            export_to_excel(
                exact_weights, IP, cloud_matrix, 
                prioritizer.weighted_matrix, prioritizer.CPIS, prioritizer.CNIS,
                prioritizer.d_plus, prioritizer.d_minus, prioritizer.ranking_scores,
                prioritizer.ranking_order, K, C, A
            )
            print(f"✓ Successfully exported all tables to Excel: results.xlsx")
        else:
            print("⚠ Skipping Excel export: Required data not available")
    except Exception as e:
        print(f"✗ Error exporting to Excel: {e}")
        import traceback
        traceback.print_exc()


def export_to_excel(exact_weights, IP, cloud_matrix, weighted_matrix, 
                    CPIS, CNIS, d_plus, d_minus, ranking_scores, ranking_order,
                    K, C, A):
    """
    Export all result tables to Excel, each in a separate sheet.
    
    Tables exported:
    - Table 3: Optimal weights (P1-P3) and Interval weight matrices (IP1-IP3)
    - Table 4: Estimations (x̄k) and Uncertainty (σk) for each DM
    - Table 5: Cloud decision matrix
    - Table 6: Weighted cloud decision matrix
    - Table 7: Ideal solutions (CPIS, CNIS)
    - Table 8: Ranking of alternatives
    """
    output_file = "results.xlsx"
    writer = pd.ExcelWriter(output_file, engine='openpyxl')
    
    # ========================================================================
    # Table 3: Optimal Weights and Interval Weight Matrices
    # ========================================================================
    for k in range(K):
        # Optimal weights (P_k)
        P_data = np.zeros((A, C))
        for j in range(C):
            for i in range(A):
                P_data[i, j] = exact_weights[k][j][i]
        
        df_P = pd.DataFrame(P_data, 
                           index=[f'A{i+1}' for i in range(A)],
                           columns=[f'C{j+1}' for j in range(C)])
        df_P.index.name = 'Alternative'
        df_P.to_excel(writer, sheet_name=f'D{k+1}_Optimal_Weights', index=True)
        
        # Interval weight matrices (IP_k)
        # Create DataFrame with lower and upper bounds in separate columns
        IP_data = []
        for i in range(A):
            row = []
            for j in range(C):
                lower = IP[k][j, i, 0]
                upper = IP[k][j, i, 1]
                row.extend([lower, upper])
            IP_data.append(row)
        
        # Column headers: C1_Lower, C1_Upper, C2_Lower, C2_Upper, ...
        IP_columns = []
        for j in range(C):
            IP_columns.extend([f'C{j+1}_Lower', f'C{j+1}_Upper'])
        
        df_IP = pd.DataFrame(IP_data,
                            index=[f'A{i+1}' for i in range(A)],
                            columns=IP_columns)
        df_IP.index.name = 'Alternative'
        df_IP.to_excel(writer, sheet_name=f'D{k+1}_Interval_Weights', index=True)
    
    # ========================================================================
    # Table 4: Estimations (x̄k) and Uncertainty (σk)
    # ========================================================================
    for k in range(K):
        # Estimations (x̄k) = (lower + upper) / 2
        x_bar_data = np.zeros((A, C))
        for j in range(C):
            for i in range(A):
                lower = IP[k][j, i, 0]
                upper = IP[k][j, i, 1]
                x_bar_data[i, j] = (lower + upper) / 2
        
        df_xbar = pd.DataFrame(x_bar_data,
                              index=[f'A{i+1}' for i in range(A)],
                              columns=[f'C{j+1}' for j in range(C)])
        df_xbar.index.name = 'Alternative'
        df_xbar.to_excel(writer, sheet_name=f'D{k+1}_Estimations', index=True)
        
        # Uncertainty (σk) = (upper - lower) / 6
        sigma_data = np.zeros((A, C))
        for j in range(C):
            for i in range(A):
                lower = IP[k][j, i, 0]
                upper = IP[k][j, i, 1]
                sigma_data[i, j] = (upper - lower) / 6
        
        df_sigma = pd.DataFrame(sigma_data,
                               index=[f'A{i+1}' for i in range(A)],
                               columns=[f'C{j+1}' for j in range(C)])
        df_sigma.index.name = 'Alternative'
        df_sigma.to_excel(writer, sheet_name=f'D{k+1}_Uncertainty', index=True)
    
    # ========================================================================
    # Table 5: Cloud Decision Matrix
    # ========================================================================
    # Format: (Ex, En, He) for each cell
    cloud_data = []
    for i in range(A):
        row = []
        for j in range(C):
            Ex = cloud_matrix[i, j, 0]
            En = cloud_matrix[i, j, 1]
            He = cloud_matrix[i, j, 2]
            row.append(f"({Ex:.4f}, {En:.4f}, {He:.4f})")
        cloud_data.append(row)
    
    df_cloud = pd.DataFrame(cloud_data,
                            index=[f'A{i+1}' for i in range(A)],
                            columns=[f'C{j+1}' for j in range(C)])
    df_cloud.index.name = 'Ỹ'
    df_cloud.to_excel(writer, sheet_name='Cloud_Decision_Matrix', index=True)
    
    # ========================================================================
    # Table 6: Weighted Cloud Decision Matrix
    # ========================================================================
    weighted_data = []
    for i in range(A):
        row = []
        for j in range(C):
            Ex = weighted_matrix[i, j, 0]
            En = weighted_matrix[i, j, 1]
            He = weighted_matrix[i, j, 2]
            row.append(f"({Ex:.4f}, {En:.4f}, {He:.4f})")
        weighted_data.append(row)
    
    df_weighted = pd.DataFrame(weighted_data,
                               index=[f'A{i+1}' for i in range(A)],
                               columns=[f'C{j+1}' for j in range(C)])
    df_weighted.index.name = 'Ŷ'
    df_weighted.to_excel(writer, sheet_name='Weighted_Cloud_Decision_Matrix', index=True)
    
    # ========================================================================
    # Table 7: Ideal Solutions
    # ========================================================================
    ideal_data = []
    for row_name, ideal_matrix in [('CPIS', CPIS), ('CNIS', CNIS)]:
        row = []
        for j in range(C):
            Ex = ideal_matrix[j, 0]
            En = ideal_matrix[j, 1]
            He = ideal_matrix[j, 2]
            row.append(f"({Ex:.4f}, {En:.4f}, {He:.4f})")
        ideal_data.append(row)
    
    df_ideal = pd.DataFrame(ideal_data,
                           index=['CPIS', 'CNIS'],
                           columns=[f'C{j+1}' for j in range(C)])
    df_ideal.to_excel(writer, sheet_name='Ideal_Solutions', index=True)
    
    # ========================================================================
    # Table 8: Ranking of Alternatives
    # ========================================================================
    ranking_data = []
    for i in range(A):
        ranking_data.append({
            'Alternatives': f'A{i+1}',
            'd_i^+': d_plus[i],
            'd_i^-': d_minus[i],
            'RS_i': ranking_scores[i],
            'Rank': np.where(ranking_order == i)[0][0] + 1
        })
    
    df_ranking = pd.DataFrame(ranking_data)
    df_ranking = df_ranking.sort_values('Rank')
    df_ranking = df_ranking[['Alternatives', 'd_i^+', 'd_i^-', 'RS_i', 'Rank']]
    df_ranking.to_excel(writer, sheet_name='Ranking_Alternatives', index=False)
    
    # Save the Excel file
    writer.close()


if __name__ == "__main__":
    main()

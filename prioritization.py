import numpy as np

class Prioritization:
    """
    Prioritization class for ranking alternatives using cloud decision matrix.
    
    Implements Algorithm 3: Prioritisation
    Steps:
    1. Construct weighted decision cloud matrix
    2. Define CPIS and CNIS
    3. Calculate distances from ideal solutions
    4. Determine ranking scores
    """
    
    def __init__(self, cloud_matrix, criteria_weights, A, C):
        """
        Parameters:
        -----------
        cloud_matrix : ndarray of shape (A, C, 3)
            Original cloud decision matrix (Ex, En, He)
        criteria_weights : ndarray of shape (C,)
            Optimal criteria weights from BiLevelOptimizer
        A : int
            Number of alternatives
        C : int
            Number of criteria
        """
        self.cloud_matrix = cloud_matrix
        self.criteria_weights = criteria_weights
        self.A = A
        self.C = C
        
        self.weighted_matrix = None
        self.CPIS = None  # Cloud Positive Ideal Solution
        self.CNIS = None  # Cloud Negative Ideal Solution
        self.d_plus = None  # Distance to CPIS
        self.d_minus = None  # Distance to CNIS
        self.ranking_scores = None
        self.ranking_order = None
    
    def _compare_clouds(self, cloud1, cloud2):
        """
        Compare two clouds using Definition 3 (H.-C. Liu et al., 2018).
        
        Parameters:
        -----------
        cloud1, cloud2 : tuple or array of (Ex, En, He)
        
        Returns:
        --------
        result : int
            -1 if cloud1 < cloud2
             0 if cloud1 = cloud2
             1 if cloud1 > cloud2
        """
        Ex1, En1, He1 = cloud1
        Ex2, En2, He2 = cloud2
        
        # Convert to interval-values
        a_bar = Ex1 + 3 * En1
        a_underline = Ex1 - 3 * En1
        b_bar = Ex2 + 3 * En2
        b_underline = Ex2 - 3 * En2
        
        # Calculate S
        S = 2 * (a_bar - b_bar) - (a_bar - a_underline + b_bar - b_underline)
        
        # Rule 1
        if S > 0:
            return 1  # cloud1 > cloud2
        elif S < 0:
            return -1  # cloud1 < cloud2
        else:
            # Rule 2
            if En1 < En2:
                return 1
            elif En1 > En2:
                return -1
            else:
                # Rule 3
                if He1 < He2:
                    return 1
                elif He1 > He2:
                    return -1
                else:
                    # Rule 4
                    return 0  # cloud1 = cloud2
    
    def _cloud_distance(self, cloud1, cloud2):
        """
        Calculate distance between two clouds using Definition 4 (Eq. 4).
        
        d(ỹ₁, ỹ₂) = sqrt(|Ex₁ - Ex₂|² + |En₁ - En₂| + |He₁ - He₂|)
        
        Parameters:
        -----------
        cloud1, cloud2 : tuple or array of (Ex, En, He)
        
        Returns:
        --------
        distance : float
        """
        Ex1, En1, He1 = cloud1
        Ex2, En2, He2 = cloud2
        
        distance = np.sqrt(
            (Ex1 - Ex2)**2 + 
            abs(En1 - En2) + 
            abs(He1 - He2)
        )
        return distance
    
    def construct_weighted_matrix(self):
        """
        Step 1: Construct weighted decision cloud matrix (Eq. 14).
        
        Ŷ_ij = Ỹ_ij × w_j for all i and j
        """
        self.weighted_matrix = np.zeros_like(self.cloud_matrix)
        
        for i in range(self.A):
            for j in range(self.C):
                # Multiply each cloud element by criterion weight
                self.weighted_matrix[i, j, 0] = self.cloud_matrix[i, j, 0] * self.criteria_weights[j]  # Ex
                self.weighted_matrix[i, j, 1] = self.cloud_matrix[i, j, 1] * self.criteria_weights[j]  # En
                self.weighted_matrix[i, j, 2] = self.cloud_matrix[i, j, 2] * self.criteria_weights[j]  # He
        
        return self.weighted_matrix
    
    def define_ideal_solutions(self):
        """
        Step 2: Define CPIS and CNIS (Eq. 16, 17).
        
        CPIS: A^+ = {max_i ỹ_ij for all j}
        CNIS: A^- = {min_i ỹ_ij for all j}
        
        Uses cloud comparison rules (Definition 3) to find max/min.
        """
        self.CPIS = np.zeros((self.C, 3))  # (C, 3) for each criterion
        self.CNIS = np.zeros((self.C, 3))
        
        for j in range(self.C):
            # Find max cloud (CPIS) for criterion j
            max_cloud_idx = 0
            for i in range(1, self.A):
                cloud_i = tuple(self.weighted_matrix[i, j, :])
                cloud_max = tuple(self.weighted_matrix[max_cloud_idx, j, :])
                if self._compare_clouds(cloud_i, cloud_max) > 0:
                    max_cloud_idx = i
            
            self.CPIS[j, :] = self.weighted_matrix[max_cloud_idx, j, :]
            
            # Find min cloud (CNIS) for criterion j
            min_cloud_idx = 0
            for i in range(1, self.A):
                cloud_i = tuple(self.weighted_matrix[i, j, :])
                cloud_min = tuple(self.weighted_matrix[min_cloud_idx, j, :])
                if self._compare_clouds(cloud_i, cloud_min) < 0:
                    min_cloud_idx = i
            
            self.CNIS[j, :] = self.weighted_matrix[min_cloud_idx, j, :]
        
        return self.CPIS, self.CNIS
    
    def calculate_distances(self):
        """
        Step 3: Calculate distances from ideal solutions (Eq. 18, 19).
        
        d_i^+ = Σ_j d(ỹ_ij, ỹ_j^+) for all i
        d_i^- = Σ_j d(ỹ_ij, ỹ_j^-) for all i
        """
        self.d_plus = np.zeros(self.A)
        self.d_minus = np.zeros(self.A)
        
        for i in range(self.A):
            d_plus_sum = 0
            d_minus_sum = 0
            
            for j in range(self.C):
                cloud_ij = tuple(self.weighted_matrix[i, j, :])
                cloud_cpis = tuple(self.CPIS[j, :])
                cloud_cnis = tuple(self.CNIS[j, :])
                
                d_plus_sum += self._cloud_distance(cloud_ij, cloud_cpis)
                d_minus_sum += self._cloud_distance(cloud_ij, cloud_cnis)
            
            self.d_plus[i] = d_plus_sum
            self.d_minus[i] = d_minus_sum
        
        return self.d_plus, self.d_minus
    
    def calculate_ranking_scores(self):
        """
        Step 4: Determine ranking scores (Eq. 20).
        
        RS_i = d_i^- / (d_i^- + d_i^+) for all i
        """
        self.ranking_scores = np.zeros(self.A)
        
        for i in range(self.A):
            denominator = self.d_minus[i] + self.d_plus[i]
            if denominator > 1e-10:  # Avoid division by zero
                self.ranking_scores[i] = self.d_minus[i] / denominator
            else:
                self.ranking_scores[i] = 0.0
        
        # Rank alternatives in descending order of RS
        self.ranking_order = np.argsort(self.ranking_scores)[::-1]
        
        return self.ranking_scores, self.ranking_order
    
    def run(self):
        """
        Execute the complete prioritization process.
        
        Returns:
        --------
        ranking_scores : ndarray
            Ranking scores for each alternative
        ranking_order : ndarray
            Indices of alternatives ranked from best to worst
        """
        self.construct_weighted_matrix()
        self.define_ideal_solutions()
        self.calculate_distances()
        self.calculate_ranking_scores()
        
        return self.ranking_scores, self.ranking_order
    
    def display_results(self):
        """Display prioritization results in a readable format."""
        if self.ranking_scores is None:
            raise ValueError("Run prioritization first. Call run() method.")
        
        print(f"\n{'='*70}")
        print("Prioritization Results")
        print(f"{'='*70}\n")
        
        print("Ranking Scores (RS_i):")
        print("-" * 70)
        for i in range(self.A):
            print(f"A{i+1}: RS = {self.ranking_scores[i]:.6f}")
        
        print(f"\n{'='*70}")
        print("Final Ranking Order (Best to Worst):")
        print(f"{'='*70}")
        for rank, alt_idx in enumerate(self.ranking_order, 1):
            print(f"Rank {rank}: A{alt_idx+1} (RS = {self.ranking_scores[alt_idx]:.6f})")
        
        print(f"\n{'='*70}")
        print("Ideal Solutions:")
        print(f"{'='*70}")
        print("\nCloud Positive Ideal Solution (CPIS):")
        print("-" * 70)
        for j in range(self.C):
            Ex, En, He = self.CPIS[j, :]
            print(f"C{j+1}: (Ex={Ex:.6f}, En={En:.6f}, He={He:.6f})")
        
        print("\nCloud Negative Ideal Solution (CNIS):")
        print("-" * 70)
        for j in range(self.C):
            Ex, En, He = self.CNIS[j, :]
            print(f"C{j+1}: (Ex={Ex:.6f}, En={En:.6f}, He={He:.6f})")
        
        print(f"\n{'='*70}")
        print("Distances from Ideal Solutions:")
        print(f"{'='*70}")
        for i in range(self.A):
            print(f"A{i+1}: d^+ = {self.d_plus[i]:.6f}, d^- = {self.d_minus[i]:.6f}")


# Usage example
if __name__ == "__main__":
    from data_loader import load_bwm_data_from_excel
    from interval_bwm import IntervalBWM_SciPy
    from cloud_dm import CloudDecisionMatrix
    from weightcal_bilevel_ooptimization import BiLevelOptimizer
    
    # Load and solve
    BO, OW, K, C, A = load_bwm_data_from_excel("data.xlsx")
    IBWM = IntervalBWM_SciPy(BO, OW, K, C, A)
    IP = IBWM.run()
    
    # Construct cloud matrix
    cdm = CloudDecisionMatrix(IP, K, C, A)
    cloud_matrix = cdm.construct()
    
    # Get criteria weights
    optimizer = BiLevelOptimizer(cloud_matrix, A, C)
    criteria_weights = optimizer.solve()
    
    # Prioritize alternatives
    prioritizer = Prioritization(cloud_matrix, criteria_weights, A, C)
    ranking_scores, ranking_order = prioritizer.run()
    prioritizer.display_results()
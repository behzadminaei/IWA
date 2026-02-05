import numpy as np

class CloudDecisionMatrix:
    """
    Constructs a Cloud Decision Matrix from interval weight matrices.
    
    Input: IP - list of interval weight matrices from multiple decision makers
           IP[k][j, i, 0] = lower bound for DM k, criterion j, alternative i
           IP[k][j, i, 1] = upper bound for DM k, criterion j, alternative i
    
    Output: Cloud decision matrix with (Ex, En, He) for each alternative-criterion pair
    """
    
    def __init__(self, IP, K, C, A):
        """
        Parameters:
        -----------
        IP : list of arrays
            Interval weight matrices from each decision maker
            IP[k] is a (C, A, 2) array
        K : int
            Number of decision makers
        C : int
            Number of criteria
        A : int
            Number of alternatives
        """
        self.IP = IP
        self.K = K  # Number of decision makers (D in the algorithm)
        self.C = C
        self.A = A
        
        self.cloud_matrix = None
    
    def _interval_to_cloud_params(self, lower, upper):
        """
        Convert interval [lower, upper] to cloud parameters (x̄, σ).
        
        x̄ = (lower + upper) / 2
        σ = (upper - lower) / 6
        """
        x_bar = (lower + upper) / 2
        sigma = (upper - lower) / 6
        return x_bar, sigma
    
    def construct(self):
        """
        Construct the cloud decision matrix following Algorithm 2.
        
        Returns:
        --------
        cloud_matrix : ndarray of shape (A, C, 3)
            Cloud decision matrix where [:, :, 0] = Ex
                                        [:, :, 1] = En
                                        [:, :, 2] = He
        """
        # Initialize cloud parameters for each DM
        # x_bar[k][j][i] and sigma[k][j][i]
        x_bar = np.zeros((self.K, self.C, self.A))
        sigma = np.zeros((self.K, self.C, self.A))
        
        # Step 1-8: Convert intervals to cloud parameters for each DM
        for k in range(self.K):
            for j in range(self.C):
                for i in range(self.A):
                    lower = self.IP[k][j, i, 0]
                    upper = self.IP[k][j, i, 1]
                    x_bar[k, j, i], sigma[k, j, i] = self._interval_to_cloud_params(lower, upper)
        
        # Initialize aggregated cloud matrix (A × C × 3)
        # Each element stores (Ex, En, He)
        cloud_matrix = np.zeros((self.A, self.C, 3))
        
        # Step 9-16: Aggregate clouds across decision makers
        for i in range(self.A):
            for j in range(self.C):
                # Gather all DMs' cloud parameters for this (alternative, criterion)
                x_bars_ij = x_bar[:, j, i]  # Shape: (K,)
                sigmas_ij = sigma[:, j, i]  # Shape: (K,)
                
                # Calculate Ex: mean of means
                Ex_ij = np.mean(x_bars_ij)
                
                # Calculate En: entropy
                mean_sigma = np.mean(sigmas_ij)
                variance_of_means = np.mean((x_bars_ij - Ex_ij) ** 2)
                En_ij = mean_sigma + np.sqrt(variance_of_means)
                
                # Calculate He: hyper-entropy
                He_ij = np.sqrt(np.mean((sigmas_ij - En_ij) ** 2))
                
                # Store in cloud matrix
                cloud_matrix[i, j, 0] = Ex_ij
                cloud_matrix[i, j, 1] = En_ij
                cloud_matrix[i, j, 2] = He_ij
        
        self.cloud_matrix = cloud_matrix
        return cloud_matrix
    
    def get_cloud_matrix(self):
        """
        Get the constructed cloud decision matrix.
        """
        if self.cloud_matrix is None:
            raise ValueError("Cloud matrix not constructed yet. Call construct() first.")
        return self.cloud_matrix
    
    def display(self):
        """
        Display the cloud decision matrix in a readable format.
        """
        if self.cloud_matrix is None:
            raise ValueError("Cloud matrix not constructed yet. Call construct() first.")
        
        print(f"\n{'='*70}")
        print("Cloud Decision Matrix")
        print(f"{'='*70}")
        
        for i in range(self.A):
            print(f"\nAlternative A{i+1}:")
            for j in range(self.C):
                Ex = self.cloud_matrix[i, j, 0]
                En = self.cloud_matrix[i, j, 1]
                He = self.cloud_matrix[i, j, 2]
                print(f"  C{j+1}: (Ex={Ex:.6f}, En={En:.6f}, He={He:.6f})")


# Usage example
if __name__ == "__main__":
    from data_loader import load_bwm_data_from_excel
    from interval_bwm import IntervalBWM_SciPy
    
    # Load data and solve BWM
    BO, OW, K, C, A = load_bwm_data_from_excel("data.xlsx")
    IBWM = IntervalBWM_SciPy(BO, OW, K, C, A)
    IP = IBWM.run()
    
    # Construct cloud decision matrix
    cdm = CloudDecisionMatrix(IP, K, C, A)
    cloud_matrix = cdm.construct()
    cdm.display()
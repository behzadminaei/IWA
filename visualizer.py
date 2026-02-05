import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class BWMVisualizer:
    """
    Visualization class for BWM analysis results.
    
    Provides three types of visualizations:
    1. Interval weights (error bars)
    2. Gaussian membership functions for decision makers
    3. Aggregated cloud droplets
    """
    
    def __init__(self, IP=None, cloud_matrix=None, K=None, C=None, A=None):
        """
        Parameters:
        -----------
        IP : list of arrays (optional)
            Interval weight matrices from each decision maker
        cloud_matrix : ndarray (optional)
            Aggregated cloud decision matrix (A × C × 3)
        K : int
            Number of decision makers
        C : int
            Number of criteria
        A : int
            Number of alternatives
        """
        self.IP = IP
        self.cloud_matrix = cloud_matrix
        self.K = K
        self.C = C
        self.A = A
    
    def plot_interval_weights(self, save_path=None):
        """
        Plot 1: Extracted interval weights with error bars for each DM.
        Similar to Fig. 2 in the paper.
        """
        if self.IP is None:
            raise ValueError("IP (interval weight matrices) not provided.")
        
        fig, axes = plt.subplots(self.K, self.C, figsize=(4*self.C, 3*self.K))
        
        # Handle single DM or single criterion case
        if self.K == 1 and self.C == 1:
            axes = np.array([[axes]])
        elif self.K == 1:
            axes = axes.reshape(1, -1)
        elif self.C == 1:
            axes = axes.reshape(-1, 1)
        
        for k in range(self.K):
            for j in range(self.C):
                ax = axes[k, j]
                
                # Extract data for this DM and criterion
                alternatives = np.arange(1, self.A + 1)
                means = []
                errors = []
                
                for i in range(self.A):
                    lower = self.IP[k][j, i, 0]
                    upper = self.IP[k][j, i, 1]
                    
                    # Ensure lower <= upper (swap if needed)
                    if lower > upper:
                        lower, upper = upper, lower
                    
                    mean = (lower + upper) / 2
                    error = abs(upper - lower) / 2  # Use abs to ensure non-negative
                    
                    means.append(mean)
                    errors.append(error)
                
                # Plot with error bars
                ax.errorbar(alternatives, means, yerr=errors, 
                           fmt='o', capsize=5, capthick=2, 
                           markersize=4, color='tab:red', 
                           ecolor='tab:blue', elinewidth=1)
                
                # Formatting
                ax.set_ylim([0, 1])
                ax.set_xlabel('Alternatives', fontsize=10)
                ax.set_xticks(alternatives)
                ax.grid(True, alpha=0.3)
                
                # Add titles
                if k == 0:
                    ax.set_title(f'C{j+1}', fontsize=12, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(f'D{k+1}', fontsize=12, fontweight='bold', rotation=0, labelpad=20)
        
        # Add main title
        fig.suptitle('Extracted Interval Weights', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_gaussian_membership_functions(self, save_path=None):
        """
        Plot 2: Gaussian membership functions for each decision maker.
        Shows probability distributions using (x̄, σ) parameters.
        Similar to Fig. 3 in the paper.
        """
        if self.IP is None:
            raise ValueError("IP (interval weight matrices) not provided.")
        
        fig, axes = plt.subplots(self.C, self.A, figsize=(4*self.A, 3*self.C))
        
        # Handle single case
        if self.C == 1 and self.A == 1:
            axes = np.array([[axes]])
        elif self.C == 1:
            axes = axes.reshape(1, -1)
        elif self.A == 1:
            axes = axes.reshape(-1, 1)
        
        # Generate x values for plotting
        x = np.linspace(0, 1, 1000)
        
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        
        for j in range(self.C):
            for i in range(self.A):
                ax = axes[j, i]
                
                # Plot Gaussian for each decision maker
                for k in range(self.K):
                    lower = self.IP[k][j, i, 0]
                    upper = self.IP[k][j, i, 1]
                    
                    # Calculate mean and std
                    x_bar = (lower + upper) / 2
                    sigma = (upper - lower) / 6
                    
                    # Gaussian function
                    if sigma > 1e-6:  # Avoid division by zero
                        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - x_bar) / sigma) ** 2)
                        # Normalize to [0, 1]
                        y = y / y.max() if y.max() > 0 else y
                    else:
                        # Delta function approximation
                        y = np.zeros_like(x)
                        idx = np.argmin(np.abs(x - x_bar))
                        y[idx] = 1.0
                    
                    ax.plot(x, y, label=f'DM{k+1}', color=colors[k % len(colors)], linewidth=1)
                
                # Formatting
                ax.set_ylim([0, 1])
                ax.set_xlim([0, 1])
                ax.grid(True, alpha=0.3)
                
                # Add titles
                if j == 0:
                    ax.set_title(f'A{i+1}', fontsize=12, fontweight='bold')
                if i == 0:
                    ax.set_ylabel(f'C{j+1}', fontsize=12, fontweight='bold', rotation=0, labelpad=20)
                
                # Add legend only to first subplot
                if j == 0 and i == self.A - 1:
                    ax.legend(loc='upper right', fontsize=8)
        
        fig.suptitle('Gaussian Membership Functions (GMF) for Decision-Makers', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cloud_droplets(self, n_drops=1000, save_path=None):
        """
        Plot 3: Aggregated cloud decision matrix with droplets.
        Generates cloud drops concentrated around Ex with spread based on En and He.
        """
        if self.cloud_matrix is None:
            raise ValueError("Cloud matrix not provided.")
        
        fig, axes = plt.subplots(self.C, self.A, figsize=(4*self.A, 3*self.C))
        
        # Handle single case
        if self.C == 1 and self.A == 1:
            axes = np.array([[axes]])
        elif self.C == 1:
            axes = axes.reshape(1, -1)
        elif self.A == 1:
            axes = axes.reshape(-1, 1)
        
        for j in range(self.C):
            for i in range(self.A):
                ax = axes[j, i]
                
                # Get cloud parameters
                Ex = self.cloud_matrix[i, j, 0]
                En = self.cloud_matrix[i, j, 1]
                He = self.cloud_matrix[i, j, 2]
                
                # Generate cloud drops
                drops_x = []
                drops_y = []
                
                for _ in range(n_drops):
                    # Step 2: Generate En' ~ N(En, He²)
                    if He > 1e-6:
                        En_prime = np.random.normal(En, He)
                        En_prime = max(En_prime, 1e-6)
                    else:
                        En_prime = En
                    
                    # Step 3: Generate x ~ N(Ex, En'²)
                    drop = np.random.normal(Ex, En_prime)
                    drop = np.clip(drop, 0, 1)
                    
                    # Step 4: Calculate certainty degree using Eq. (3) with En' (Hadi and Uwe Paper)
                    if En_prime > 1e-6:
                        membership = np.exp(-(drop - Ex)**2 / (2 * En_prime**2))
                    else:
                        membership = 1.0 if abs(drop - Ex) < 1e-6 else 0.0
                    
                    drops_x.append(drop)
                    drops_y.append(membership)
                
                # Plot drops
                ax.scatter(drops_x, drops_y, alpha=0.6, s=10, color='tab:blue', edgecolors='none')
                
                # Formatting
                ax.set_ylim([0, 1])
                ax.set_xlim([0, 1])
                ax.grid(True, alpha=0.3)
                
                # Add titles
                if j == 0:
                    ax.set_title(f'A{i+1}', fontsize=12, fontweight='bold')
                if i == 0:
                    ax.set_ylabel(f'C{j+1}', fontsize=12, fontweight='bold', rotation=0, labelpad=20)
        
        fig.suptitle('Aggregated Cloud Decision Matrix', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_all(self, save_prefix=None):
        """
        Generate all three plots.
        
        Parameters:
        -----------
        save_prefix : str (optional)
            Prefix for saving files. Will create:
            - {prefix}_interval_weights.png
            - {prefix}_gmf.png
            - {prefix}_cloud_droplets.png
        """
        print("Generating visualizations...")
        
        if self.IP is not None:
            print("1/3: Plotting interval weights...")
            save_path = f"{save_prefix}_interval_weights.png" if save_prefix else None
            self.plot_interval_weights(save_path)
        
        if self.IP is not None:
            print("2/3: Plotting Gaussian membership functions...")
            save_path = f"{save_prefix}_gmf.png" if save_prefix else None
            self.plot_gaussian_membership_functions(save_path)
        
        if self.cloud_matrix is not None:
            print("3/3: Plotting cloud droplets...")
            save_path = f"{save_prefix}_cloud_droplets.png" if save_prefix else None
            self.plot_cloud_droplets(n_drops=500, save_path=save_path)
        
        print("Visualization complete!")


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
    
    # Visualize
    viz = BWMVisualizer(IP=IP, cloud_matrix=cloud_matrix, K=K, C=C, A=A)
    viz.plot_all(save_prefix="bwm_results")
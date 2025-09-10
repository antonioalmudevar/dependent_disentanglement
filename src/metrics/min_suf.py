import numpy as np
from scipy.stats import entropy

class InfoEstimator:
    """
    A class to estimate mutual information and entropy from discrete and continuous data.
    """

    def __init__(self, y_discrete, z_continuous):
        """
        Initializes the estimator with discrete and continuous data.
        
        Args:
            y_discrete (np.ndarray): An n x m numpy array of discrete realizations.
            z_continuous (np.ndarray): An n x p numpy array of continuous realizations.
        """
        if y_discrete.ndim != 2 or z_continuous.ndim != 2:
            raise ValueError("Input arrays must be 2D (n x features).")
        if y_discrete.shape[0] != z_continuous.shape[0]:
            raise ValueError("The number of realizations (n) must be the same for both arrays.")

        self.y_discrete = y_discrete
        self.z_continuous = z_continuous
        self.n_samples, self.m_features = y_discrete.shape
        _, self.p_features = z_continuous.shape
        self.z_discrete = None

    def discretize_z(self, n_bins=10):
        """
        Discretizes the continuous vector z using uniform binning.
        
        Args:
            n_bins (int): The number of bins to use for discretization.
        """
        self.n_bins = n_bins
        self.z_discrete = np.empty_like(self.z_continuous, dtype=int)
        for i in range(self.p_features):
            z_col = self.z_continuous[:, i]
            # Use np.digitize to bin the data
            bins = np.linspace(np.min(z_col), np.max(z_col), n_bins + 1)
            # Adjust the upper bound to ensure max value is included
            bins[-1] += 1e-9 
            self.z_discrete[:, i] = np.digitize(z_col, bins) - 1


    def _get_counts(self, data):
        """
        Helper function to get the probability mass function (PMF) from discrete data.
        """
        # Count occurrences of each unique value
        unique, counts = np.unique(data, return_counts=True)
        # Normalize to get probabilities (PMF)
        probabilities = counts / np.sum(counts)
        return probabilities

    def calculate_entropy(self):
        """
        Estimates the entropy for each component of y (discrete) and z (discretized).
        
        Returns:
            dict: A dictionary containing 'H(Y)' and 'H(Z)' arrays.
        """
        if self.z_discrete is None:
            raise RuntimeError("z must be discretized first. Call discretize_z().")
        
        entropy_y = np.zeros(self.m_features)
        entropy_z = np.zeros(self.p_features)

        # Entropy for y components
        for i in range(self.m_features):
            pmf = self._get_counts(self.y_discrete[:, i])
            entropy_y[i] = entropy(pmf, base=2) # Base 2 for bits
        
        # Entropy for z components
        for i in range(self.p_features):
            pmf = self._get_counts(self.z_discrete[:, i])
            entropy_z[i] = entropy(pmf, base=2)
            
        return entropy_y, entropy_z

    def calculate_mutual_information(self):
        """
        Estimates the mutual information I(Y_i; Z_j) for all pairs.
        
        Returns:
            np.ndarray: A m x p matrix where the element at (i, j) is I(Y_i; Z_j).
        """
        if self.z_discrete is None:
            raise RuntimeError("z must be discretized first. Call discretize_z().")
            
        mutual_info_matrix = np.zeros((self.m_features, self.p_features))

        for i in range(self.m_features):
            for j in range(self.p_features):
                y_col = self.y_discrete[:, i]
                z_col = self.z_discrete[:, j]
                
                # Calculate joint PMF for (Y_i, Z_j)
                joint_counts = np.zeros((self.y_discrete.max() + 1, self.n_bins))
                for y_val, z_val in zip(y_col, z_col):
                    joint_counts[y_val, z_val] += 1
                
                # Normalize to get joint probabilities
                joint_pmf = joint_counts / np.sum(joint_counts)
                
                # Calculate marginal PMFs
                pmf_y = self._get_counts(y_col)
                pmf_z = self._get_counts(z_col)
                
                # Calculate entropies
                H_y = entropy(pmf_y, base=2)
                H_z = entropy(pmf_z, base=2)
                
                # Calculate joint entropy H(Y_i, Z_j)
                H_yz = entropy(joint_pmf.flatten(), base=2)
                
                # Mutual Information formula: I(Y;Z) = H(Y) + H(Z) - H(Y,Z)
                mutual_info = H_y + H_z - H_yz
                mutual_info_matrix[i, j] = mutual_info
                
        return mutual_info_matrix


def estimate_min_suf(y, z, n_bins=15):
    y = y.astype(int)
    estimator = InfoEstimator(y, z)
    estimator.discretize_z(n_bins=n_bins)
    ent_y, ent_z = estimator.calculate_entropy()
    mutual_info = estimator.calculate_mutual_information()
    minimality = (mutual_info.max(axis=0) / ent_z).mean()
    sufficiency = (mutual_info.max(axis=1) / ent_y).mean()
    return minimality, sufficiency
import numpy as np
import spatial_dp as sd
from spatial_dp import spatial_depth

def integrated_ld(data, dm0, nbr_list=None, depth='spatial', beta_list=None, Lmatrix=None):
    """
    Integrated local depth (ILD) calculation, spatial-depth only.
    
    Args:
        data: (n, d) data matrix
        dm0: (n, n) depth-based (possibly non-symmetric) similarity matrix
        nbr_list: list of neighborhood sizes to compute ILD (overrides beta_list if given)
        depth: which depth to use, default 'spatial'
        beta_list: if given, nbr_list = ceil(beta_list * n)
        Lmatrix: (optional) distance matrix for efficient spatial depth (as in R code)
        
    Returns:
        dict with:
            - ld.mat: (n, len(nbr_list)) raw local depth matrix
            - dm0.order: (n, n) neighbor ordering by decreasing similarity (each column = neighbors for that point)
            - ILD_mat: (n, len(nbr_list)) integrated local depth (cumulative mean)
            - nbr_list: array of neighbor sizes used
    """
    n, d = data.shape
    if beta_list is not None and len(beta_list) > 0:
        nbr_list = np.ceil(np.array(beta_list) * n).astype(int)
    if nbr_list is None or len(nbr_list) == 0:
        start_point = min(2 * d, 10)
        nbr_list = np.arange(start_point, n + 1)
    b = len(nbr_list)
    ld_mat = np.zeros((n, b))

    # Each column j: indices of neighbors for point j, sorted by decreasing similarity
    # R: dm0.order <- sapply(1:n, function(i) { sort.list(dm0[i,], decreasing = T) })
    # This creates a matrix where each column contains neighbor indices for that point
    dm0_order = np.argsort(-dm0, axis=1).T  # Transpose to match R: each column = neighbors for that point

    if Lmatrix is None:
        for idx, i in enumerate(nbr_list):
            # For each locality/neighbor size
            if i != n:
                for j in range(n):
                    sub_idx = dm0_order[:i, j]
                    # Only spatial depth implemented (Mahalanobis skipped)
                    ld_mat[j, idx] = spatial_depth(data[sub_idx[0]], data[sub_idx])
            else:
                    ld_mat[:, idx] = spatial_depth(data, data)
    else:
        tol = 1e-5
        Lmatrix = Lmatrix.copy()
        Lmatrix[Lmatrix < tol] = np.inf
        Lmatrix = 1 / Lmatrix
        for j in range(n):
            label = dm0_order[:max(nbr_list), j]
            Lmatrix_sub = Lmatrix[j, label]
            norm_csum = np.cumsum(Lmatrix_sub)
            norm_csum = norm_csum[[i - 1 for i in nbr_list]]
            C2 = np.outer(norm_csum, data[j])
            C_all = data[label] * Lmatrix_sub[:, None]
            cumsum_matrix = np.cumsum(C_all, axis=0)
            C = cumsum_matrix[[i - 1 for i in nbr_list], :]
            C = C2 - C
            mean_C = C / nbr_list[:, None]
            ld_mat[j, :] = 1 - np.sqrt(np.sum(mean_C ** 2, axis=1))
    
    # Integrated local depth: for each point, cumulative mean of local depths up to each neighborhood size
    ILD_mat = np.array([np.cumsum(row) / (np.arange(len(row)) + 1) for row in ld_mat])

    return {
        "ld.mat": ld_mat,           # Local depth matrix (n × b)
        "dm0.order": dm0_order,     # Neighbor ordering (n × n)
        "ILD_mat": ILD_mat,         # Integrated local depth (n × b)
        "nbr_list": nbr_list        # Neighborhood sizes used (length b)
    }
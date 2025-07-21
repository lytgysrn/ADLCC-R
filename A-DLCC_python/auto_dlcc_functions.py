import numpy as np
from typing import Dict, List,  Optional
from spatial_dp import spatial_depth
from sklearn.ensemble import RandomForestClassifier
import copy
from scipy.stats import rankdata
# Placeholder functions for step-by-step implementation

# Note: sym_mat is imported from ADLCC module, so we don't redefine it here



def find_peak_size(ILD_mat, dm0_order, nbr_list, min_nbr_th=19):
    """
    Identify valid local centers via integrated local depth (ILD).

    Args:
        ILD_mat: (n_points, n_levels) integrated local depth matrix
        dm0_order: (n_points, n_points) neighbor order indices (descending similarity, 0-based)
                   each column contains neighbor indices for that point
        nbr_list: list/array of neighbor sizes (length n_levels)
        min_nbr_th: int, minimum number of locality levels for a valid ILD (default )
    Returns:
        dict with:
            - highest_ld_idx: list of length n_points, each is array of highest-ILD neighbors across all levels
            - est_size: array of length n_points, estimated neighborhood size for each point (0 if not a local center)
    """
    n_points, n_levels = ILD_mat.shape
    # Ranking matrix: for each locality level (column), rank ILD values (descending)
  
    ranking_mat = np.apply_along_axis(lambda x: rankdata(-x, method='ordinal'), 0, ILD_mat)

    highest_ld_idx = []
    est_size = np.zeros(n_points, dtype=int) 
    
    for i in range(n_points):
        ild_row = ILD_mat[i, :]
        highest_ld = np.zeros(n_levels, dtype=int)
        
        for j in range(len(nbr_list)):  
            k = nbr_list[j]
            # Get the neighbor indices for point i at local level j (first k neighbors)
            nbrs = dm0_order[:k, i]           # indices of top-k neighbors for point i (from column i)
            candidate_ranks = ranking_mat[nbrs, j]
            # the one with highest ild (lowest rank since we ranked -x)
            best_neighbor = nbrs[np.argmin(candidate_ranks)]
            highest_ld[j] = best_neighbor
            
        # save result
        highest_ld_idx.append(highest_ld)

        # Find neighborhood levels where point i is the local center
        own_max_idx = np.where(highest_ld == i)[0]
        # Only consider those with enough neighbors (matching R logic)
        own_max_idx = own_max_idx[own_max_idx > min_nbr_th]
        
        lidx = len(own_max_idx)
        if lidx >= 2:
            # Use the level with the highest ILD value
            pos_idx = own_max_idx[np.argmax(ild_row[own_max_idx])]
            est_size[i] = nbr_list[pos_idx]
            
    return {
        "highest_ld_idx": highest_ld_idx,  # list of np.arrays
        "est_size": est_size               # np.array, length n_points
    }

def check_lc(est_size, highest_ld_idx, nbr_list, dm0_order):
    """
    Filter local centers by frequency (proportion) in local neighborhoods.

    Args:
        est_size: (n_points,) array, estimated size for each point (0 if not a center)
        highest_ld_idx: list of arrays, for each point the highest-ILD neighbor indices at each level
        nbr_list: list/array of neighborhood sizes
        dm0_order: (n_points, n_points) neighbor order indices
                   each column contains neighbor indices for that point
    Returns:
        dict with:
            - save_lc: array of retained local center indices (sorted by frequency)
            - save_size: array of corresponding estimated sizes
            - freq_table: frequency table (dict: point_idx -> frequency, descending)
    """
    current_lc = np.where(est_size != 0)[0]
    save_lc = current_lc.tolist()
    save_size = est_size[est_size != 0].tolist()

    # Check frequency property for each candidate local center
    for i in current_lc:
        s = est_size[i]
        pos_s = np.where(nbr_list == s)[0]
        if len(pos_s) == 0:
            continue
        pos_s = pos_s[0]
        
        # For all points, does i appear in their top-s neighbors?
        # Equivalent to R: cols_with_tg <- apply(dm0.order[1:s, ], 2, function(col) any(col == i))
        cols_with_tg = [np.any(dm0_order[:s, col] == i) for col in range(dm0_order.shape[1])]
        
        # Equivalent to R: result_vector <- sapply(highest_ld_idx[cols_with_tg], function(sublist) sublist[pos_s])
        result_vector = [highest_ld_idx[col][pos_s] for col, flag in enumerate(cols_with_tg) if flag]
        
        # Equivalent to R: own_prop = sum(result_vector==i)/length(result_vector)
        own_prop = np.sum(np.array(result_vector) == i) / len(result_vector) if result_vector else 0
        
        if own_prop < 0.5:
            idx = save_lc.index(i)
            save_size.pop(idx)
            save_lc.pop(idx)

    # Build frequency table (equivalent to R: freq_table = table(unlist(highest_ld_idx)))
    flat_list = np.concatenate(highest_ld_idx)
    unique, counts = np.unique(flat_list, return_counts=True)
    freq_table = dict(zip(unique, counts))
    
    # Sort by frequency descending (equivalent to R: sort(decreasing = T))
    freq_table_sorted = dict(sorted(freq_table.items(), key=lambda item: item[1], reverse=True))

    # Reorder save_lc and save_size according to frequency table order
    # Equivalent to R: sorted_idx = order(match(save_lc %>% as.character(), names(freq_table)))
    lc_str = list(map(str, save_lc))
    freq_keys_str = list(map(str, freq_table_sorted.keys()))
    sorted_idx = [lc_str.index(x) for x in freq_keys_str if x in lc_str]
    sorted_save_lc = [save_lc[i] for i in sorted_idx]
    sorted_save_size = [save_size[i] for i in sorted_idx]

    return {
        "save_lc": np.array(sorted_save_lc, dtype=int),
        "save_size": np.array(sorted_save_size, dtype=int),
        "freq_table": freq_table_sorted
    }

def sym_mat(dm0):
    """
    Convert a depth matrix to a symmetric matrix.
    """
    return (dm0 + dm0.T) / 2


def sm_computer(save_lc: np.ndarray, sym_dm: np.ndarray) -> Dict:
    """
    Compute similarity matrix information.
    
    Args:
        save_lc: local centers
        sym_dm: symmetric similarity matrix for points
        
    Returns:
        dict with similarity matrix information
    """
    # G_1 to G_T: assign each point to the local center with highest similarity
    assignments = np.array([save_lc[np.argmax(sym_dm[i, save_lc])] for i in range(sym_dm.shape[0])])
    
    # Count assignments for each local center
    unique, counts = np.unique(assignments, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    # Sort counts by save_lc order
    sort_counts = np.array([count_dict.get(lc, 0) for lc in save_lc])
    
    # Check if any group contains only 1 point, if exists drop and update
    while np.any(sort_counts == 1):
        # Remove local centers with only 1 point
        save_lc = save_lc[sort_counts != 1]
        
        # Reassign points
        assignments = np.array([save_lc[np.argmax(sym_dm[i, save_lc])] for i in range(sym_dm.shape[0])])
        
        # Recalculate counts
        unique, counts = np.unique(assignments, return_counts=True)
        count_dict = dict(zip(unique, counts))
        sort_counts = np.array([count_dict.get(lc, 0) for lc in save_lc])
    
    n_lc = len(save_lc)
    r = np.zeros(n_lc)
    nbr_save = []
    
    # Create neighbor lists for each local center
    for i in range(n_lc):
        own = save_lc[i]
        nbr = np.where(assignments == own)[0]
        nbr_save.append(nbr)
    
    # Create TF matrix to reduce computation
    TFmatrix = np.zeros((n_lc, n_lc), dtype=bool)
    
    # Check group relationships
    # R: for (g in seq_along(nbr_save)) { ... }
    for g in range(len(nbr_save)):
        own_pt = save_lc[g]
        pts = nbr_save[g]
        
        # Calculate inter-group minimum and intra-group maximum
        # R: inter_min = sym_dm[pts,pts] %>% min()
        inter_min = np.min(sym_dm[np.ix_(pts, pts)])
        
        # R: intra_max = sapply(nbr_save[-g], function(x) { sym_dm[own_pt, x] %>% max() })
        intra_max = []
        for other_group in nbr_save[:g] + nbr_save[g+1:]:
            intra_max.append(np.max(sym_dm[own_pt, other_group]))
        
        # R: rec_flag <- intra_max >= inter_min
        rec_flag = np.array(intra_max) >= inter_min
        
        # Set TF matrix values
        # R: TFmatrix[g, -g] <- rec_flag
        other_indices = list(range(g)) + list(range(g+1, n_lc))
        TFmatrix[g, other_indices] = rec_flag
    
    # Make TF matrix symmetric
    # R: TFmatrix <- pmax(TFmatrix, t(TFmatrix))
    TFmatrix = np.maximum(TFmatrix, TFmatrix.T)
    
    # Generate similarity matrix
    sym_sm = sim_matrix_generator(sym_dm, assignments, save_lc, TFmatrix, nbr_save, sort_counts)
    
    return {
        'sym_sm': sym_sm,
        'counts': sort_counts,
        'TFmatrix': TFmatrix,
        'nbr_save': nbr_save,
        'save_lc': save_lc
    }

def sim_matrix_generator(d: np.ndarray, assignments: np.ndarray, save_lc: np.ndarray, 
                        TFmatrix: np.ndarray, nbr_save: list, sort_counts: np.ndarray) -> np.ndarray:
    """
    Generate similarity matrix using GLS computation.
    
    Args:
        d: similarity matrix
        assignments: group assignments
        save_lc: local centers
        TFmatrix: indicator matrix for computing similarity between groups
        nbr_save: neighbor lists for each group
        sort_counts: counts for each group
        
    Returns:
        symmetric normalized similarity matrix
    """
    N = d.shape[0]
    n = len(save_lc)
    
    # Initialize similarity matrix
    c = np.zeros((n, n))
    
    # Find upper triangular indices where TFmatrix == 1
    upper_tri = np.triu(np.ones_like(TFmatrix), k=1)
    upper_idx = np.where((TFmatrix == 1) & upper_tri)
    upper_idx = list(zip(upper_idx[0], upper_idx[1]))
    
    # Process each pair of groups
    for group_i, group_j in upper_idx:
        for x in nbr_save[group_i]:
            for y in nbr_save[group_j]:
                # Get depth similarities
                dx = d[x, :]
                dy = d[y, :]
                assignment_y = assignments[y]
                assignment_x = assignments[x]
                
                # Find points in Uxy
                uxy = np.where((dx >= d[x, y]) | (dy >= d[y, x]))[0]
                uxy_x = uxy[assignments[uxy] == assignment_x]
                uxy_y = uxy[assignments[uxy] == assignment_y]
                
                # Calculate weights
                wx = 1.0 * (dx[uxy_x] > dy[uxy_x]) + 0.5 * (dx[uxy_x] == dy[uxy_x])
                
                wy = 1.0 * (dx[uxy_y] > dy[uxy_y]) + 0.5 * (dx[uxy_y] == dy[uxy_y])
                
                pos_x = group_i
                pos_y = group_j
                
                # Update similarity matrix
                
                c[pos_x, pos_y] += np.sum(wy) / len(uxy_y)
                
                # R: c[pos_y,pos_x] = c[pos_y,pos_x] + sum(1-wx)/length(uxy_x)
                c[pos_y, pos_x] += np.sum(1 - wx) / len(uxy_x)
    
    # Normalize similarity matrix
    # R: normalized_c <- t(sapply(1:nrow(c), function(i) { c[i,] / (sort_counts * sort_counts[i]) }))
    normalized_c = np.zeros_like(c)
    for i in range(c.shape[0]):
        normalized_c[i, :] = c[i, :] / (sort_counts * sort_counts[i])
    
    # Make symmetric
    sym_normalized_c = sym_mat(normalized_c)
    
    return sym_normalized_c

def reachable_similarity(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the reachable similarity matrix (bottleneck path similarity).
    For each pair of nodes, the value is the maximum bottleneck similarity along any path connecting them.

    Args:
        sim_matrix: (n, n) similarity matrix

    Returns:
        R: (n, n) reachable similarity matrix
    """
    n = sim_matrix.shape[0]
    if n == 2:
        return sim_matrix.copy()
    else:
        # Initialize result matrix with 0, diagonal with 1
        R = np.zeros((n, n), dtype=sim_matrix.dtype)
        np.fill_diagonal(R, 1)

        # Only consider edges with weight > 0 (no connection if <= 0)
        # Get upper triangle unique pairs
        edges = np.transpose(np.triu_indices(n, k=1))
        edge_weights = sim_matrix[edges[:, 0], edges[:, 1]]
        valid = edge_weights > 0
        edges = edges[valid]
        edge_weights = edge_weights[valid]

        # Sort edges by weight descending
        sorted_idx = np.argsort(-edge_weights)
        edges = edges[sorted_idx]
        edge_weights = edge_weights[sorted_idx]

        # Each node starts in its own component
        comp = np.arange(n)
        for e in range(len(edges)):
            u, v = edges[e]
            w = edge_weights[e]
            if comp[u] != comp[v]:
                cu = comp[u]
                cv = comp[v]
                nodes_u = np.where(comp == cu)[0]
                nodes_v = np.where(comp == cv)[0]
                # Update reachable similarity for all new connections
                for i in nodes_u:
                    for j in nodes_v:
                        R[i, j] = max(R[i, j], w)
                        R[j, i] = max(R[j, i], w)
                # Merge the components: relabel component cv as cu
                comp[comp == cv] = cu
        return R


def LNI_strategy_choice(sym_sm: np.ndarray, R: np.ndarray) -> dict:
    """
    Choose clustering strategy ('centroid' or 'linkage') based on similarity matrices.

    Args:
        sym_sm: symmetric similarity matrix
        R: reachable similarity matrix

    Returns:
        dict with keys: 'prop', 'strategy', 'ambiguity'
    """
    d = R.shape[0]
    R = R.copy()
    sym_sm = sym_sm.copy()
    np.fill_diagonal(R, 0)
    np.fill_diagonal(sym_sm, 0)
    value = R - sym_sm

    # prop: proportion of reachable similarity that is not explained by sym_sm
    prop = np.sum(value[sym_sm < 0.01]) / np.sum(R)
    ambiguity = False

    if prop <= 0.4:
        strategy = 'centroid'
    elif prop >= 0.6:
        strategy = 'linkage'
    else:
        R0_prop = (np.sum(R < 0.01) - d) / (np.sum(sym_sm < 0.01) - d)
        if R0_prop > 0.5:
            strategy = 'linkage'
        else:
            strategy = 'centroid'
        ambiguity = True

    return {'prop': prop, 'strategy': strategy, 'ambiguity': ambiguity}

def ifmerging(dm0_order: np.ndarray, nbr_save: list, save_lc: np.ndarray, N: int) -> bool:
    """
    Decide whether further merging is needed based on neighbor structure.

    Args:
        dm0_order: (n, n) neighbor order matrix (each column: neighbors for that point, 0-based)
        nbr_save: list of arrays, each contains indices of points in a group
        save_lc: array of local center indices
        N: number of local centers

    Returns:
        need_merging: bool, whether further merging is needed
    """
    if N == 2:
        return False
    else:
        max_nbr = max(len(nbr) for nbr in nbr_save)
        size = round(max_nbr / 2)

        need_merging = False
        for i, idx in enumerate(save_lc):
            # R: nbrs <- dm0.order[2:size, idx]
            # Python: rows 1 to size-1 (since Python is 0-based, skip the first neighbor)
            if size > 1:
                nbrs = dm0_order[1:size, idx]
            else:
                nbrs = np.array([], dtype=int)
            # if any other exemplars in its neighbors, set True
            other_exemplars = set(save_lc) - {idx}
            if np.any(np.isin(nbrs, list(other_exemplars))):
                need_merging = True
                break
        return need_merging
            
def find_stable_centers_ld(
    matrix_info: dict,
    sym_dm: np.ndarray,
    strategy: Optional[list] = None,
    dm0_order: Optional[np.ndarray] = None) -> dict:
    """
    Find stable centers using local depth.
    Python implementation aligned with R code logic.
    """
    save_lc = matrix_info['save_lc'].copy()
    sym_sm = matrix_info['sym_sm'].copy()
    nbr_save = copy.deepcopy(matrix_info['nbr_save'])

    N = len(save_lc)

    # Reachable similarity
    R = reachable_similarity(sym_sm)
    np.fill_diagonal(sym_sm, 1)

    # Strategy selection
    if not strategy or len(strategy) == 0:
        LNI_result = LNI_strategy_choice(sym_sm, R)
        strategy = LNI_result['strategy']
        if LNI_result.get('ambiguity', False):
            print(f"Warning: p is {LNI_result['prop']:.3f}, which is in the ambiguous range (0.4 ~ 0.6). You may try the other strategy to see the difference.")

    # Reachable list
    reachable_list = []
    for x in nbr_save:
        rsm = reachable_similarity(sym_dm[np.ix_(x, x)])
        cs = np.sum(rsm, axis=0)
        reachable_list.append((cs - 1) / (len(x) - 1))

    if strategy == 'linkage':
        group_list = adaptive_merge(save_lc, R, sym_dm, sym_sm, nbr_save, reachable_list)
        return {'group_list': group_list, 'strategy': 'linkage'}
    elif strategy == 'centroid':
        merging = ifmerging(dm0_order, nbr_save, save_lc, N) if dm0_order is not None else True
        if merging:
            group_info = adaptive_flex_merge(save_lc, R, sym_dm, sym_sm, nbr_save, reachable_list)
        else:
            group_info = {'group_list': [np.array([lc]) for lc in save_lc], 'temp_clus': nbr_save}
        return {'group_list': group_info, 'strategy': 'centroid'}
    elif strategy == 'both':
        # Run both strategies
        linkage_result = adaptive_merge(save_lc, R, sym_dm, sym_sm, nbr_save, reachable_list)
        
        merging = ifmerging(dm0_order, nbr_save, save_lc, N) if dm0_order is not None else True
        if merging:
            centroid_result = adaptive_flex_merge(save_lc, R, sym_dm, sym_sm, nbr_save, reachable_list)
        else:
            centroid_result = {'group_list': [np.array([lc]) for lc in save_lc], 'temp_clus': nbr_save}
        
        return {
            'linkage_res': linkage_result,
            'centroid_res': centroid_result,
            'strategy': 'both'
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Expected 'linkage', 'centroid', or 'both'")



def update_index(input_idx, drop_idx):
    """
    Update indices after dropping some groups.
    """
    input_idx = np.array(input_idx)
    drop_idx = np.array(drop_idx, ndmin=1)
    # For each x in input_idx, if x in drop_idx return 0, else x - sum(drop_idx < x)
    return np.array([
        0 if x in drop_idx else x - np.sum(drop_idx < x)
        for x in input_idx
    ])

def update_group_similarity(group_list, sym_sm):
    """
    Update the group similarity matrix based on new group_list.
    Each entry (i, j) is the max similarity between any member of group i and any member of group j.
    """
    K = len(group_list)
    group_sim = np.zeros((K, K))
    for i in range(K - 1):
        for j in range(i + 1, K):
            group_i = group_list[i]
            group_j = group_list[j]
            max_sim = np.max(sym_sm[np.ix_(group_i, group_j)])
            group_sim[i, j] = max_sim
            group_sim[j, i] = max_sim
    return group_sim

def compute_mcs(after_mer):
    """
    Compute intra-group similarity.
    """
    cs = np.sum(after_mer, axis=0)
    mcs = (cs - 1) / (after_mer.shape[0] - 1)
    return mcs


def check_pairs(pairs, sym_sm, group_list, min_close):
    """
    Accept singularly most similar pairing.
    Args:
        pairs: list of [from, to] index pairs
        sym_sm: similarity matrix
        group_list: list of lists of indices (groups)
        min_close: float, minimum similarity threshold
    Returns:
        filtered pairs (list)
    """
    keep = np.zeros(len(pairs), dtype=bool)
    for i, pair in enumerate(pairs):
        from_idx = [idx for idx in pair if len(group_list[idx]) == 1]
        to_idx = [idx for idx in pair if idx not in from_idx]
        if to_idx:
            from_to_sim = sym_sm[np.ix_(group_list[from_idx[0]], group_list[to_idx[0]])]
            ftc = np.mean(from_to_sim)
            tc = np.min(compute_mcs(sym_sm[np.ix_(group_list[to_idx[0]], group_list[to_idx[0]])]))
            # tot = np.max(sym_sm[np.ix_(group_list[to_idx[0]], np.setdiff1d(np.arange(sym_sm.shape[0]), group_list[to_idx[0]]))], axis=1)
            judge1 = np.all(from_to_sim > min_close)
            judge2 = ftc >= tc
            if judge1 or judge2:
                keep[i] = True
    return [pair for i, pair in enumerate(pairs) if keep[i]]

def calc_pro(smaller_group, larger_group, result_clusters, group_list, sym_dm, radius):
    """
    Delta computing (linkage).
    Args:
        smaller_group: int, group index
        larger_group: int, group index
        result_clusters: list of arrays (indices)
        group_list: list of arrays (indices)
        sym_dm: similarity matrix
        radius: float
    Returns:
        pro: float
    """
    counts_small = len(result_clusters[smaller_group])
    counts_large = len(result_clusters[larger_group])
    to_larger = np.max(sym_dm[np.ix_(result_clusters[smaller_group], result_clusters[larger_group])], axis=1)
    lower_bound = np.max(sym_dm[np.ix_(group_list[smaller_group], group_list[larger_group])])
    if lower_bound >= radius:
        pro = np.sum(to_larger >= radius) / counts_small
    else:
        p_value = counts_small / counts_large
        dynamic_radius = lower_bound * (1 - np.sqrt(p_value)) + radius * np.sqrt(p_value)
        pro = np.sum(to_larger >= dynamic_radius) / counts_small
    return pro

def cnbr_generator(temp_sm, singleton_group=None):
    """
    Find mutually most similar pairs.
    Args:
        temp_sm: similarity matrix
        singleton_group: list of indices (optional)
    Returns:
        pairs: list of [i, j] pairs
    """
    n = temp_sm.shape[0]
    pairs = []
    if singleton_group is None or len(singleton_group) == 0:
        row_max_idx = np.argmax(temp_sm, axis=1)
        for i in range(n):
            j = row_max_idx[i]
            if i < j and row_max_idx[j] == i:
                pairs.append([i, j])
    else:
        from_list, to_list, sim_list = [], [], []
        for i in singleton_group:
            sims = temp_sm[i, :].copy()
            sims[i] = -np.inf
            j = np.argmax(sims)
            from_list.append(i)
            to_list.append(j)
            sim_list.append(sims[j])
        df = sorted(zip(from_list, to_list, sim_list), key=lambda x: -x[2])
        used = np.zeros(n, dtype=bool)
        result = []
        for i, j, _ in df:
            if not used[i] and not used[j]:
                result.append([i, j])
                used[i] = used[j] = True
        pairs = result
    return pairs

def adaptive_merge(save_lc, R, sym_dm, sym_sm, nbr_save, reachable_list):
    """
    adaptive merging （linkage)
    """
    all_mer = reachable_similarity(sym_dm)
    global_th = np.quantile(np.mean(all_mer, axis=1), 0.05)
    N = len(save_lc)
    # intra-group similarity
    current_mcs = copy.deepcopy(reachable_list)
    result_clusters = copy.deepcopy(nbr_save)
    row_mean_to = []
    for x in result_clusters:
        mask = np.ones(all_mer.shape[0], dtype=bool)
        mask[x] = False
        row_mean_to.append(np.mean(all_mer[np.ix_(x, mask)], axis=1))
    bth_all = np.array([np.mean(row_mean_to[x] / current_mcs[x]) for x in range(N)])
    bth_all = np.clip(bth_all, 0.9, 0.99)
    R_active = (sym_sm != 0).astype(int)
    np.fill_diagonal(R_active, 0)
    group_list = [[i] for i in range(N)]
    temp_sm = sym_sm.copy()
    np.fill_diagonal(temp_sm, 0)
    accept_save = [[] for _ in range(N)]

    # first stage
    # find the mutually most similar pairs
    pairs = cnbr_generator(temp_sm)
    # same group min similarity
    ps = [sym_sm[x[0], x[1]] for x in pairs] 
    th_fs = min(ps) 
    min_pos = np.min(sym_sm[sym_sm != 0])
    
    while True:
  
        if not pairs:
            break
        to_merge_list = []
        mcs_save = []
        Q_save = []
        for merge_target in pairs:
            merge_ids = np.concatenate([result_clusters[idx] for idx in merge_target])
            after_mer = reachable_similarity(sym_dm[np.ix_(merge_ids, merge_ids)])
            mcs = compute_mcs(after_mer)
            delta = np.zeros(len(merge_target))
            for i, idx in enumerate(merge_target):
                g_points = result_clusters[idx]
                g_mean = current_mcs[idx]
                pos_idx = [np.where(merge_ids == gp)[0][0] for gp in g_points]
                to_other = np.mean(after_mer[np.ix_(pos_idx, np.setdiff1d(np.arange(len(merge_ids)), pos_idx))], axis=1)
                delta[i] = np.mean(to_other / g_mean)
            Q_min = np.min(delta)
            Q_th = np.min(bth_all[merge_target])
            if Q_min > Q_th:
                to_merge_list.append(merge_target)
                mcs_save.append(mcs)
                Q_save.append(Q_min)
            else:
                R_active[np.ix_(merge_target, merge_target)]  = 0
                
        n_m = len(to_merge_list)
        if n_m > 0:
            drop_idx = []
            for i in range(n_m):
                merge_target = to_merge_list[i]
                change_pos = min(merge_target)
                drop_pos = [idx for idx in merge_target if idx != change_pos]
                drop_idx.extend(drop_pos)
                # Save original content of change_pos before merging
                merge_ids = np.concatenate([result_clusters[idx] for idx in merge_target])
                result_clusters[change_pos] = merge_ids
                current_mcs[change_pos] = mcs_save[i]
                #update base threshold, rmt should be computed as row-wise mean to other points, like row_mean_to 
                mask = np.ones(all_mer.shape[0], dtype=bool)
                mask[result_clusters[change_pos]] = False
                rmt = np.mean(all_mer[np.ix_(result_clusters[change_pos], mask)], axis=1)
                bth_value = max(0.9, min(0.99, np.mean(rmt / mcs_save[i])))
                bth_all[change_pos] = bth_value
                group_list[change_pos] = sum([group_list[idx] for idx in merge_target], [])
                value = np.sum(R_active[merge_target, :], axis=0) > 0
                R_active[change_pos, :] = value
                R_active[:, change_pos] = value
                np.fill_diagonal(R_active, 0)
                accept_save[change_pos] = accept_save[change_pos] + sum([accept_save[idx] for idx in merge_target if idx != change_pos], []) + [Q_save[i]]
            if drop_idx:
                keep_idx = [i for i in range(len(result_clusters)) if i not in drop_idx]
                R_active = R_active[np.ix_(keep_idx, keep_idx)]
                result_clusters = [result_clusters[i] for i in keep_idx]
                group_list = [group_list[i] for i in keep_idx]
                current_mcs = [current_mcs[i] for i in keep_idx]
                accept_save = [accept_save[i] for i in keep_idx]
                bth_all = bth_all[keep_idx]
        temp_sm = update_group_similarity(group_list, sym_sm)
        temp_sm = temp_sm * (R_active != 0)
        pairs = cnbr_generator(temp_sm)
        sim_pairs = [temp_sm[x[0], x[1]] for x in pairs]
        idx_accept = [v >= th_fs for v in sim_pairs]
        if any(idx_accept):
            pairs = [p for p, accept in zip(pairs, idx_accept) if accept]
        else:
            lg = [len(g) for g in group_list]
            singleton_group = [i for i, l in enumerate(lg) if l == 1]
            if singleton_group:
                pairs = cnbr_generator(temp_sm, singleton_group)
                pairs = check_pairs(pairs, sym_sm, group_list, min_close=min_pos)
            else:
                pairs = [p for p, accept in zip(pairs, idx_accept) if accept]
            if pairs:
                th_fs = min([
                    np.min(np.max(sym_sm[np.ix_(indices, indices)] * (1 - np.eye(len(indices))), axis=1))
                    for indices in group_list if len(indices) > 1
                ])
 
    while np.max(R_active) > 0:

        temp_sm = update_group_similarity(group_list, sym_sm)
        temp_sm = temp_sm * (R_active != 0)
        mts = np.max(temp_sm)
        if abs(mts) > 1e-10:
            all_pairs = cnbr_generator(temp_sm)
            involve_point = sorted(set([idx for pair in all_pairs for idx in pair]))
            base = min(involve_point, key=lambda x: len(result_clusters[x]))
            merge_target = sorted(set([idx for pair in all_pairs if base in pair for idx in pair]))
            merge_ids = np.concatenate([result_clusters[idx] for idx in merge_target])
            after_mer = reachable_similarity(sym_dm[np.ix_(merge_ids, merge_ids)])
            mcs = compute_mcs(after_mer)
            delta = np.zeros(len(merge_target))
            prop_worse = np.zeros(len(merge_target))
            for i, idx in enumerate(merge_target):
                g_points = result_clusters[idx]
                g_mean = current_mcs[idx]
                pos_idx = [np.where(merge_ids == gp)[0][0] for gp in g_points]
                to_other = np.mean(after_mer[np.ix_(pos_idx, np.setdiff1d(np.arange(len(merge_ids)), pos_idx))], axis=1)
                delta[i] = np.mean(to_other / g_mean)
                prop_worse[i] = np.sum(mcs[pos_idx] < g_mean - 1e-10) / len(g_points)
            Q_min = np.min(delta)
            bth = np.min(bth_all[merge_target])
            prop_worse_mean = np.mean(prop_worse)
            accept_vals = sum([accept_save[idx] for idx in merge_target], [])
            median_accept = np.median(accept_vals) if accept_vals else 1.0
            median_accept = min(float(median_accept), 1.0)
            if Q_min < median_accept:
                cluster_size = [len(result_clusters[idx]) for idx in merge_target]
                smaller_group = merge_target[np.argmin(cluster_size)]
                larger_group = merge_target[np.argmax(cluster_size)]
                radius = np.mean([global_th, np.quantile(mcs, 0.05)])
                gap_prop = 1 - calc_pro(smaller_group, larger_group, result_clusters, group_list, sym_dm, radius)
                Q_th = bth + gap_prop * prop_worse_mean ** 2 * (median_accept - bth)
            else:
                Q_th = bth
            if Q_min > Q_th:
                change_pos = min(merge_target)
                drop_idx = [idx for idx in merge_target if idx != change_pos]
                # Save original content of change_pos before merging
                merge_ids = np.concatenate([result_clusters[idx] for idx in merge_target])
                result_clusters[change_pos] = merge_ids
                current_mcs[change_pos] = mcs
                mask = np.ones(all_mer.shape[0], dtype=bool)
                mask[result_clusters[change_pos]] = False
                rmt = np.mean(all_mer[np.ix_(result_clusters[change_pos], mask)], axis=1)
                bth_value = max(0.9, min(0.99, np.mean(rmt / mcs)))
                bth_all[change_pos] = bth_value
                group_list[change_pos] = sum([group_list[idx] for idx in merge_target], [])
                value = np.sum(R_active[merge_target, :], axis=0) > 0
                R_active[change_pos, :] = value
                R_active[:, change_pos] = value
                np.fill_diagonal(R_active, 0)
                accept_save[change_pos] = accept_save[change_pos] + sum([accept_save[idx] for idx in merge_target if idx != change_pos], []) + [Q_min]
                keep_idx = [i for i in range(len(result_clusters)) if i not in drop_idx]
                R_active = R_active[np.ix_(keep_idx, keep_idx)]
                result_clusters = [result_clusters[i] for i in keep_idx]
                group_list = [group_list[i] for i in keep_idx]
                current_mcs = [current_mcs[i] for i in keep_idx]
                accept_save = [accept_save[i] for i in keep_idx]
                bth_all = bth_all[keep_idx]
            else:
                R_active[np.ix_(merge_target, merge_target)]  = 0
        else:
            R_active[R_active != 0] = 0
   
    group_list_final = [save_lc[np.array(g)] for g in group_list if len(g) > 0]
    return {'group_list': group_list_final, 'temp_clus': result_clusters}



def adaptive_flex_merge(save_lc, R, sym_dm, sym_sm, nbr_save, reachable_list):
    """
    adaptive merging （centroid)
    """

    all_mer = reachable_similarity(sym_dm)
    overall_max = np.max(sym_sm * np.triu(np.ones_like(sym_sm), 1))
    # valid_neighbors_list: for each group, sorted similarities to other groups (excluding self)
    valid_neighbors_list = []
    for x in range(len(save_lc)):
        other_indices = [i for i in range(len(save_lc)) if i != x and sym_sm[x, i] != 0]
        sim_values = sym_sm[x, other_indices]
        # descending order
        sorted_idx = np.argsort(sim_values)[::-1]
        # Map back to original save_lc indices
        sorted_labels = [save_lc[other_indices[i]] for i in sorted_idx]
        sorted_sim_values = sim_values[sorted_idx]
        valid_neighbors_list.append((sorted_labels, sorted_sim_values))

    #group_list = [[x] for x in range(len(save_lc))]

    # row_mean_to: for each group, mean reachable similarity to all other groups
    row_mean_to = []
    for x in nbr_save:
        mask = np.ones(all_mer.shape[0], dtype=bool)
        mask[x] = False
        row_mean_to.append(np.mean(all_mer[np.ix_(x, mask)], axis=1))
    bth_all = np.array([np.mean(row_mean_to[x] / reachable_list[x]) for x in range(len(save_lc))])

    drop_idx_base = np.where(bth_all > 1)[0]
    if len(drop_idx_base) > 0:
        dl_mat = (sym_sm == R) & (sym_sm != 0)
        mean_reachable = np.array([np.mean(r) for r in reachable_list])
        est_size = np.array([len(n) for n in nbr_save])
        m_size = np.median(est_size)

        is_drop = []
        for i in drop_idx_base:
            direct_nbr = [j for j in np.where(dl_mat[i])[0] if j not in drop_idx_base]
            own_reach = mean_reachable[i]
            if len(direct_nbr) > 0:
                nbr_reach = mean_reachable[direct_nbr]
                is_drop.append(np.min(nbr_reach) > own_reach)
            else:
                is_drop.append(est_size[i] < m_size)
        drop_idx = drop_idx_base[is_drop]
    else:
        drop_idx = np.array([], dtype=int)

    if (len(save_lc) - len(drop_idx)) > 1:
        if len(drop_idx) > 0:
            # Remove dropped groups
            save_lc = np.delete(save_lc, drop_idx)
            valid_neighbors_list = [v for i, v in enumerate(valid_neighbors_list) if i not in drop_idx]
            sym_sm = np.delete(np.delete(sym_sm, drop_idx, axis=0), drop_idx, axis=1)
            nbr_save = [n for i, n in enumerate(nbr_save) if i not in drop_idx]
            reachable_list = [r for i, r in enumerate(reachable_list) if i not in drop_idx]
            R = np.delete(np.delete(R, drop_idx, axis=0), drop_idx, axis=1)

            left_obs = np.concatenate(nbr_save)
            all_mer_sub = reachable_similarity(sym_dm[np.ix_(left_obs, left_obs)])
            row_mean_to = []
            for x in nbr_save:
                idx = np.array([np.where(left_obs == xi)[0][0] for xi in x])
                mask = np.ones(len(left_obs), dtype=bool)
                mask[idx] = False
                row_mean_to.append(np.mean(all_mer_sub[np.ix_(idx, mask)], axis=1))
            bth_all = np.array([np.mean(row_mean_to[x] / reachable_list[x]) for x in range(len(save_lc))])

    bth_all = np.clip(bth_all, 0.9, 0.99)
    N = len(save_lc)
    if N > 2:
        accept_save = [[] for _ in range(N)]
        considered_nbr_list = [[] for _ in range(N)]
        bound = 1
        stop_bound = 0.9 * np.min(bth_all)
        for i in range(N):
            sort_sim_labels, sort_sim_values = valid_neighbors_list[i]
            # Find positions of labels that are less than current index i
            # We need to map back from save_lc values to indices
            pos = []
            for j, label in enumerate(sort_sim_labels):
                # Find the index of this label in save_lc
                label_matches = np.where(save_lc == label)[0]
                if len(label_matches) > 0:
                    label_idx = label_matches[0]
                    if label_idx < i:
                        pos.append(label_idx)
            sim_values = np.delete(sym_sm[i, :], i)
            min_sim = np.min(sim_values)
            gap_sim = overall_max - min_sim
            if len(pos) > 0:
                l_pos = len(pos)
                Q = 1
                j = 0
                while Q > stop_bound and j < l_pos:
                    idx = pos[j]
                    sv = sym_sm[i, idx]
                    merge_target = [i, idx]
                    merge_ids = np.concatenate([nbr_save[m] for m in merge_target])
                    after_mer = reachable_similarity(sym_dm[np.ix_(merge_ids, merge_ids)])
                    mcs = compute_mcs(after_mer)
                    delta = np.zeros(len(merge_target))
                    prop_worse = np.zeros(len(merge_target))
                    for u, v in enumerate(merge_target):
                        g_points = nbr_save[v]
                        g_mean = reachable_list[v]
                        pos_idx = []
                        for gp in g_points:
                            matches = np.where(merge_ids == gp)[0]
                            if len(matches) > 0:
                                pos_idx.append(matches[0])
                        pos_idx = np.array(pos_idx)
                        if len(pos_idx) > 0:
                            mask = np.ones(len(merge_ids), dtype=bool)
                            mask[pos_idx] = False
                            to_other = np.mean(after_mer[np.ix_(pos_idx, mask)], axis=1)
                            delta[u] = np.mean(to_other / g_mean)
                            prop_worse[u] = np.sum(mcs[pos_idx] < g_mean - 1e-10) / len(g_points)
                        else:
                            delta[u] = 1.0  # Default value if no matches found
                            prop_worse[u] = 0.0
                    Q = np.min(delta)
                    Q_base = np.min(bth_all[merge_target])
                    if Q_base < bound:
                        prop_worse_mean = np.mean(prop_worse)
                        gap_prop = 1 - (sv - min_sim) / gap_sim if abs(gap_sim) > 1e-10 else 0
                        Q_th = Q_base + gap_prop * prop_worse_mean ** 2 * (bound - Q_base)
                    else:
                        Q_th = Q_base
                    if Q > Q_th:
                        considered_nbr_list[i].append(idx)
                        accept_save[i].append(Q)
                        if Q < bound:
                            bound = Q
                    j += 1
        ifstable = [len(c) == 0 for c in considered_nbr_list]
        if sum(ifstable) == 1:
            accept_value = [max(x) if len(x) > 0 else 1 for x in accept_save]
            ifstable[np.argmin(accept_value)] = True
    else:
        ifstable = [True] * N

    # Assign groups
    group = []
    for x in range(len(save_lc)):
        row_R = R[x, ifstable] * sym_sm[x, ifstable]
        candidate_idx = np.where(row_R == np.max(row_R))[0]
        if len(candidate_idx) == 1:
            group.append(candidate_idx[0])
        else:
            row_sim = R[x, ifstable][candidate_idx]
            chosen = candidate_idx[np.argmax(row_sim)]
            group.append(chosen)

    # Build group_list and result_clusters
    group_list = [[] for _ in range(max(group) + 1)]
    for idx, g in enumerate(group):
        group_list[g].append(idx)
    
    max_indices = np.argmax(sym_dm[:, save_lc], axis=1)  
    result_clusters = []
    for group in group_list:  
        cluster_points = []
        for i in group:
            cluster_points.extend(np.where(max_indices == i)[0])
        result_clusters.append(np.array(cluster_points))
    

    est_num = [len(rc) for rc in result_clusters]
    low_size = np.median(est_num) / 2
    reconsider = [i for i, n in enumerate(est_num) if n < low_size]
    lrec = len(reconsider)
    lgap = len(group_list) - lrec

    # Check for small size group and merge if needed
    if lrec > 0 and lgap > 1:
        reachable_list = [
        (np.sum(reachable_similarity(sym_dm[np.ix_(x, x)]), axis=0) - 1) / (len(x) - 1)
        for x in result_clusters]

        row_mean_to = []
        for x in result_clusters:
            mask = np.ones(all_mer.shape[0], dtype=bool)
            mask[x] = False
            row_mean_to.append(np.mean(all_mer[np.ix_(x, mask)], axis=1))
        bth_all = np.array([np.mean(row_mean_to[x] / reachable_list[x]) for x in range(len(result_clusters))])

        max_num = max(est_num)
        weighted_R = sym_sm * R
        temp_sm = update_group_similarity(group_list, weighted_R)
        if len(reconsider) > 1:
            row_max_idx = np.argmax(temp_sm[reconsider, :], axis=1)
        else:
            row_max_idx = [np.argmax(temp_sm[reconsider[0], :])]
        pair_list = []
        for i in range(len(reconsider)):
            pair = sorted([int(reconsider[i]), int(row_max_idx[i])])
            pair_list.append(pair)
        # Remove duplicates
        pair_list = [list(x) for x in set(tuple(x) for x in pair_list)]
        num_point = [sum([est_num[idx] for idx in pair]) for pair in pair_list]
        pair_list = [pair for pair, n in zip(pair_list, num_point) if n <= max_num]
        while len(pair_list) > 0:
            merge_target = pair_list[0]
            sizes = [len(result_clusters[idx]) for idx in merge_target if result_clusters[idx] is not None]
            size_sum = sum(sizes)
            merge_ids = np.concatenate([result_clusters[idx] for idx in merge_target if result_clusters[idx] is not None]).astype(int)
            after_mer = reachable_similarity(sym_dm[np.ix_(merge_ids, merge_ids)])
            mcs = compute_mcs(after_mer)
            valid_indices = [i for i in range(all_mer.shape[0]) if i not in merge_ids]
            mask = np.array(valid_indices, dtype=bool)
            current_rowmean = np.mean(all_mer[np.ix_(np.array(merge_ids), np.array(valid_indices))], axis=1)
            update_ratio = np.mean(current_rowmean / mcs)
            w1 = np.sqrt(sizes[0])
            w2 = np.sqrt(sizes[1])
            origin_ratio = w1 / (w1 + w2) * bth_all[merge_target[0]] + w2 / (w1 + w2) * bth_all[merge_target[1]]
            if update_ratio <= origin_ratio:
                left_label = min(merge_target)
                drop_label = max(merge_target)
                group_list[left_label].extend(group_list[drop_label])
                result_clusters[left_label] = np.concatenate([result_clusters[left_label], result_clusters[drop_label]]).astype(int)
                bth_all[left_label] = update_ratio
                del group_list[drop_label]
                del result_clusters[drop_label]
                pair_list = pair_list[1:]
                # update pair_list
                new_pair_list = []
                for pair in pair_list:
                    # Replace drop_label with left_label
                    pair = [left_label if x == drop_label else x for x in pair]
                    # Update indices after dropping
                    pair = update_index(pair, [drop_label])
                    new_pair_list.append(sorted(pair))
                pair_list = new_pair_list
                if size_sum < low_size:
                    temp_sm = update_group_similarity(group_list, weighted_R)
                    max_idx = int(np.argmax(temp_sm[left_label, :]))
                    append_content = [int(left_label), max_idx]
                    if sum([len(result_clusters[idx]) for idx in append_content]) <= max_num:
                        pair_list.append(append_content)
                # Remove duplicates
                pair_list = [list(x) for x in set(tuple(sorted(pair)) for pair in pair_list if len(pair) == 2)]
            else:
                pair_list = pair_list[1:]

    # Convert group_list indices back to save_lc indices
    group_list_final = [save_lc[np.array(g)] for g in group_list if len(g) > 0]

    return {'group': group, 'group_list': group_list_final, 'temp_clus': [rc for rc in result_clusters if len(rc) > 0]}

def check_point(group_list: List[np.ndarray], sym_dm: np.ndarray, assignments: np.ndarray, n: int) -> Dict:
    """
    Check retained points and construct initial temporary clusters.
    """
    K = len(group_list)
    temp_clus = [[] for _ in range(K)]
    left_clus = [[] for _ in range(K)]
    
    for i in range(K):
        a_c = group_list[i]
        lac = len(a_c)
        if lac > 1:
            nbr_ac = [np.where(assignments == x)[0] for x in a_c]
            ng_points = np.setdiff1d(np.arange(n), np.concatenate(nbr_ac))
            for j in range(lac):
                points = nbr_ac[j]
                other_points = np.concatenate([nbr_ac[k] for k in range(lac) if k != j])
                is_vals = np.max(sym_dm[np.ix_(points, other_points)], axis=1)
                bs_vals = np.max(sym_dm[np.ix_(points, ng_points)], axis=1)
                TFlabel = is_vals - bs_vals > 0
                TFlabel[points == a_c[j]] = True
                temp_clus[i].extend(points[TFlabel])
                left_clus[i].extend(points[~TFlabel])
        else:
            points = np.where(assignments == a_c[0])[0]
            ng_points = np.setdiff1d(np.arange(n), points)
            
            overlap = []
            for j in points:
                threshold = sym_dm[j, a_c[0]]
                overlap_count = np.sum(sym_dm[j, ng_points] >= threshold)
                overlap.append(overlap_count)
            
            TFlabel = np.array(overlap) == 0
            temp_clus[i] = points[TFlabel].tolist()
            left_clus[i] = points[~TFlabel].tolist()
    
    return {'temp_clus': temp_clus, 'left_clus': left_clus}

def assign_score(Kclus: int, dm0: np.ndarray, temp_clus: List[List], temp_cl: List[List]) -> List[List]:
    """
    Calculate scores for clustering.
    """
    a = np.concatenate(temp_cl)
    a=np.array(a,dtype=int)
    deflist = [[] for _ in range(Kclus)]
    
    for k in range(Kclus):
        tmp_l = len(temp_clus[k])
        if tmp_l != 0:
            max_a = []
            max_b = []
            for x in range(tmp_l):
                # Ensure temp_clus[k][x] is an integer for indexing
                idx = int(temp_clus[k][x])
                # Convert lists to arrays
                temp_cl_array = np.array(temp_cl[k], dtype=int)
                setdiff_array = np.setdiff1d(a, temp_cl[k])
                max_a.append(np.max(dm0[temp_cl_array, idx]))
                max_b.append(np.max(dm0[setdiff_array, idx]))
            
            larger = []
            for x in range(tmp_l):
                if max_a[x] - max_b[x] > 0:
                    larger.append(max_a[x])
                else:
                    larger.append(max_b[x])
            
            deflist[k] = [(max_a[x] - max_b[x]) / larger[x] for x in range(tmp_l)]
    
    return deflist

def get_temp_clus(sym_dm: np.ndarray, group_info: Dict, strategy: str, 
                 data: np.ndarray, freq_table: Optional[Dict] = None) -> Dict:
    """
    Obtain final temporary clusters.
    """
    group_list = group_info['group_list']
    tmp_clus = group_info['temp_clus']
    
    a = np.concatenate(group_list)
    lg = [len(g) for g in group_list]
    Kclus = len(group_list)
    clus_ind = list(range(Kclus))
    
    # Handle singletons
    if any(l == 1 for l in lg):
        add_pos = [i for i, l in enumerate(lg) if l == 1]
        freq_keys = list(freq_table.keys()) if freq_table else []
        candidates = [x for x in freq_keys if x not in a]
        candidates = np.array(candidates, dtype=int)
        filled_singletons = np.zeros(len(add_pos))
        
        for v_idx, v in enumerate(add_pos):
            n_can = len(candidates)
            singleton_a = group_list[v]
            to_sa = sym_dm[candidates, singleton_a]
            to_others = np.max(sym_dm[np.ix_(candidates, np.setdiff1d(a, singleton_a))], axis=1)
            ranks_idx = np.where(to_sa > to_others)[0]
            candidates_v = candidates[ranks_idx]

            depth_value = spatial_depth(data[candidates_v], data[tmp_clus[v]]) * (n_can - ranks_idx) / n_can
            
            best_point = candidates_v[np.argmax(depth_value)]
            filled_singletons[v_idx] = best_point
            
            candidates = candidates[candidates != best_point]
        
        # Update group_list with filled singletons
        for u, pos in enumerate(add_pos):
            if filled_singletons[u] != 0:
                group_list[pos] = np.append(group_list[pos], filled_singletons[u])
        
        # Update info
        a = np.concatenate(group_list)
        lg = [len(g) for g in group_list]
    
    n = sym_dm.shape[0]
    

    # Ensure 'a' is of integer type for indexing
    a_int = np.array(a, dtype=int)
    assignments = np.array([a_int[np.argmax(sym_dm[i, a_int])] for i in range(n)])
    
    temp_clus_result = check_point(group_list, sym_dm, assignments, n)
    temp_clus = temp_clus_result['temp_clus']
    left_clus = temp_clus_result['left_clus']
    
    score_temp = assign_score(Kclus, sym_dm, temp_clus, group_list)
    left_num = sum(len(lc) for lc in left_clus)
    
    if left_num > 0:
        score_temp2 = assign_score(Kclus, sym_dm, left_clus, group_list)
        left_scores = [s for sublist in score_temp2 for s in sublist]
        
        if len(left_scores) > 0:
            mean_score2 = np.mean(left_scores)
        else:
            mean_score2 = 0
        
        if strategy != 'linkage':
            lq = []
            for x in range(Kclus):
                len2 = len(score_temp2[x])
                len1 = len(score_temp[x])
                if len2 <= len1:
                    lq.append(mean_score2)
                else:
                    lq.append(0.5 * mean_score2)
        else:
            lq = [0.5 * mean_score2] * Kclus
        
        # Move points between temp_clus and left_clus
        for i in range(Kclus):
            index = [j for j, score in enumerate(score_temp[i]) if score < lq[i]]
            if len(index) > 0:
                left_clus[i].extend([temp_clus[i][j] for j in index])
                score_temp2[i].extend([score_temp[i][j] for j in index])
                temp_clus[i] = [temp_clus[i][j] for j in range(len(temp_clus[i])) if j not in index]
                score_temp[i] = [score_temp[i][j] for j in range(len(score_temp[i])) if j not in index]
        
        # Calculate new borders
        new_border = []
        ltc = [len(tc) for tc in temp_clus]
        llc = [len(lc) for lc in left_clus]
        
        for i in clus_ind:
            num_obs_current = ltc[i]
            num_total = num_obs_current + llc[i]
            
            if len(score_temp[i]) > 0:
                sortscore = np.sort(score_temp[i])[::-1]
                start_idx = max(0, num_obs_current // 2 - 1)
                end_idx = min(num_obs_current, len(sortscore))
                if end_idx > start_idx:
                    sortscore_subset = sortscore[start_idx:end_idx]
                    if len(sortscore_subset) > 1:
                        cdd_1 = sortscore_subset[np.argmin(np.diff(sortscore_subset))]
                    else:
                        cdd_1 = sortscore_subset[0] if len(sortscore_subset) > 0 else 1
                else:
                    cdd_1 = 1
            else:
                cdd_1 = 1
            
            if num_obs_current < 0.5 * num_total:
                temp_s = 1 - (0.5 * num_total - num_obs_current) / len(score_temp2[i])
                if temp_s > 0 and len(score_temp2[i]) > 0:
                    cdd_2 = np.quantile(score_temp2[i], temp_s)
                else:
                    cdd_2 = 0
            else:
                cdd_2 = 1
            
            new_border.append(min(cdd_1, cdd_2))
        
        # Update left_clus based on new borders
        for x in clus_ind:
            if len(score_temp2[x]) > 0:
                index = [j for j, score in enumerate(score_temp2[x]) if score > new_border[x]]
                left_clus[x] = [left_clus[x][j] for j in index]
        
        # Merge temp_clus and left_clus
        for i in clus_ind:
            temp_clus[i].extend(left_clus[i])
    
    return {
        'temp_clus': temp_clus,
        'group_list': group_list,
        'stable_centers': a
    }

def KNNdep(k: int, K: int, dm0: np.ndarray, classes: np.ndarray) -> Dict:
    """
    KNN classification using depth-based similarity matrix.
    Args:
        k: the number of clusters
        K: K for the knn algorithm (# of nbr to use)
        dm0: n*m depth-based similarity matrix, n represents # of points for        classes: known classes of points (cluster assignments)
        
    Returns:
        dict with keys: 'Dmatrix', 'class'
        - Dmatrix: (D, k) matrix where Dmatrix[i,j] is count of class j in k nearest neighbors of point i
        - class: (D,) array of predicted class assignments
    """
    D = dm0.shape[0]
    Dmatrix = np.zeros((D, k), dtype=int)
    
    # For each point j, find k nearest neighbors and count their classes
    for j in range(D):
        d = dm0[j, :]  
        nearest_indices = np.argsort(d)[::-1][:K]
        nearest_classes = classes[nearest_indices]
        
        # Count occurrences of each class in the k nearest neighbors
        for i in range(k):
             Dmatrix[j, i] = np.sum(nearest_classes == (i + 1)) 
    
    # Assign class based on majority vote with tie-breaking
    Class_med = np.zeros(D, dtype=int)
    for x in range(D):
        # Find classes with maximum count
        max_count = np.max(Dmatrix[x, :])
        result = np.where(Dmatrix[x, :] == max_count)[0] + 1  
        
        lr = len(result)
        
        # Tie breaker: based on the most similar neighbor
        if lr > 1:
            maxsim2clus = []
            for i in range(lr):
                # Find points belonging to class result[i]
                class_points = np.where(classes == result[i])[0]
                if len(class_points) > 0:
                    # Get maximum similarity to any point in this class
                    max_sim = np.max(dm0[x, class_points])
                    maxsim2clus.append(max_sim)
                else:
                    maxsim2clus.append(0)
            
            # Choose class with highest maximum similarity
            max_index = np.argmax(maxsim2clus)
            result = result[max_index]
        else:
            result = result[0]  # Single result
        
        Class_med[x] = result
    
    return {'Dmatrix': Dmatrix, 'class': Class_med}

def cluster2cv(data: np.ndarray, temp_clus: List[List]) -> np.ndarray:
    """
    Convert cluster assignments to label vector.
    
    Args:
        data: Data matrix (n, d)
        temp_clus: List of temporary clusters, each containing point indices
        
    Returns:
        current_label: Array of cluster assignments (0 for unassigned, 1-K for clusters)
    """
    current_label = np.zeros(data.shape[0], dtype=int)
    for i, clus in enumerate(temp_clus):
        current_label[clus] = i + 1
    return current_label


def DAobs(data: np.ndarray, temp_clus: List[List], method: str = 'knn', 
          depth: str = 'spatial', ntrees: int = 100,
          K_knn: int = 7, dm0: Optional[np.ndarray] = None, leaf_size: int = 0) -> Dict:
    """
    Assign observations to clusters using various methods.
    
    Args:
        data: Data matrix (n, d)
        temp_clus: List of temporary clusters
        method: Classification method ('knn', 'maxdep', 'rf')
        depth: Depth method ('spatial' only)
        ntrees: Number of trees for random forest (if method='rf')
        K_knn: Number of neighbors for KNN (if method='knn')
        dm0: Depth-based similarity matrix (required for method='knn')
        
    Returns:
        dict with keys: 'cluster', 'cluster_vector'
        - cluster: List of clusters, each containing point indices
        - cluster_vector: Array of cluster assignments for each point
    """
    if depth != 'spatial':
        raise ValueError("Currently only spatial depth is supported")
    
    X = data
    d = X.shape[1]
    Nobs = X.shape[0]
    Kclus = len(temp_clus)
    
    # Get labeled observations (points in temp_clus)
    labelledobs = np.concatenate(temp_clus)
    left_obs = X[np.setdiff1d(np.arange(Nobs), labelledobs)]
    left_obs_label = np.setdiff1d(np.arange(Nobs), labelledobs)
    N_left = len(left_obs_label)
    
    if N_left > 0:
        if method == 'maxdep':
            # Maximum depth method
            depth_mat = np.zeros((left_obs.shape[0], Kclus))
            for i in range(left_obs.shape[0]):
                for j in range(Kclus):
                    if depth == 'spatial':
                        depth_mat[i, j] = spatial_depth(left_obs[i:i+1], X[temp_clus[j]])
            
            # Sort depths in descending order for each point
            depth_order = np.argsort(depth_mat, axis=1)[:, ::-1]
            
            # Assign clusters based on maximum depth
            cluster = []
            for i in range(Kclus):
                cluster_i = left_obs_label[depth_order[:, 0] == i]
                cluster.append(np.concatenate([temp_clus[i], cluster_i]))
            
            # Create cluster vector
            cc = cluster2cv(X, cluster)
                
        elif method == 'rf':
            current_label = cluster2cv(X, temp_clus)
            # Get labeled data for training
            labeled_mask = current_label != 0
            N_labelled = np.sum(labeled_mask)
            
            # Build random forest model with dynamic leaf size
            # Set leaf_size proportional to N to avoid overfitting
            # if leaf_size == 0:
            #     leaf_size = int(np.ceil(0.0025 * N_labelled))
            #     if leaf_size > 10:
            #         leaf_size = 10
            
            # Train Random Forest classifier
            rf_classifier = RandomForestClassifier(
                n_estimators=ntrees, 
                min_samples_leaf=3, 
                bootstrap=True  # Bag method equivalent
            )
            rf_classifier.fit(X[labeled_mask], current_label[labeled_mask])
            
            # Predict for unlabeled points
            unlabeled_mask = current_label == 0
            if np.any(unlabeled_mask):
                pred = rf_classifier.predict(X[unlabeled_mask])
                current_label[unlabeled_mask] = pred
            
            cc = current_label
            
            # Create cluster list
            cluster = []
            for i in range(Kclus):
                cluster_i = left_obs_label[current_label[left_obs_label] == i + 1]
                cluster.append(np.concatenate([temp_clus[i], cluster_i]))
                
        elif method == 'knn':
            
            if dm0 is None:
                raise ValueError("dm0 (depth-based similarity matrix) is required for KNN method")
            
            current_label = cluster2cv(X, temp_clus)
            labeled_indices = np.where(current_label != 0)[0]
            classes = current_label[labeled_indices]
            unlabeled_indices = np.where(current_label == 0)[0]
            
            # Extract the relevant subset of dm0: similarities from unlabeled to labeled points
            dm0_subset = dm0[np.ix_(unlabeled_indices, labeled_indices)]
            
            # KNNdep expects dm0 to be (n_unlabeled, n_labeled) and classes to be (n_labeled,)
            knn_result = KNNdep(k=Kclus, K=K_knn, dm0=dm0_subset, classes=classes)
            current_label[unlabeled_indices] = knn_result['class']
            
            cc = current_label
            
            # Create cluster list
            cluster = []
            for i in range(Kclus):
                cluster_i = left_obs_label[knn_result['class'] == i + 1]
                cluster.append(np.concatenate([temp_clus[i], cluster_i]))
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'knn', 'maxdep', or 'rf'")
    else:
        # No unlabeled observations
        cluster = temp_clus
        cc = cluster2cv(X, temp_clus)
    
    return {'cluster': cluster, 'cluster_vector': cc}


def AUTO_DLCC(ILD_info: Dict, dm0: np.ndarray, data: np.ndarray, 
              class_method: str = 'knn', K_knn: int = 7, 
              depth: str = 'spatial', strategy: Optional[List[str]] = None,
              ntrees: int = 100) -> Dict:
    """
    AUTO_DLCC function for automatic clustering using depth-based methods.
    
    Args:
        ILD_info: Dictionary containing ILD_mat, dm0.order, nbr_list
        dm0: Depth-based similarity matrix
        data: Data matrix (n, d)
        class_method: Classification method ('knn', 'maxdep', 'rf')
        K_knn: Number of neighbors for KNN
        depth: Depth method ('spatial')
        strategy: Strategy to use ('linkage', 'centroid', 'both') or None for auto-selection
        ntrees: Number of trees for random forest (if method='rf')
        
    Returns:
        Dictionary containing clustering results:
        - If strategy is 'linkage' or 'centroid': standard result format
        - If strategy is 'both': contains 'linkage_res' and 'centroid_res' keys
    """
    ILD_mat = ILD_info['ILD_mat']
    dm0_order = ILD_info['dm0.order']
    nbr_list = ILD_info['nbr_list']
    
    # Convert dm0 to symmetric matrix
    sym_dm = sym_mat(dm0)
    
    # Find peak size information
    size_info = find_peak_size(ILD_mat, dm0_order, nbr_list)
    est_size = size_info['est_size']
    highest_ld_idx = size_info['highest_ld_idx']
    
    # Check local consistency
    save_lc_contents = check_lc(est_size, highest_ld_idx, nbr_list, dm0_order)
    save_lc = save_lc_contents['save_lc']
    freq_table = save_lc_contents['freq_table']
    
    # Compute similarity matrix
    matrix_info = sm_computer(save_lc, sym_dm)
    
    # Find stable centers
    stable_centers_info = find_stable_centers_ld(
        matrix_info=matrix_info, 
        sym_dm=sym_dm, 
        strategy=strategy, 
        dm0_order=dm0_order
    )
    
    if stable_centers_info['strategy'] == 'both':
        # Handle both strategies
        linkage_group_info = stable_centers_info['linkage_res']
        centroid_group_info = stable_centers_info['centroid_res']
        
        # Get temporary clustering for linkage
        linkage_temp_clus = get_temp_clus(
            sym_dm, 
            linkage_group_info, 
            strategy='linkage', 
            data=data, 
            freq_table=freq_table
        )
        
        # Get temporary clustering for centroid
        centroid_temp_clus = get_temp_clus(
            sym_dm, 
            centroid_group_info, 
            strategy='centroid', 
            data=data, 
            freq_table=freq_table
        )
        
        # Final clustering assignment for linkage
        linkage_cluster_result = DAobs(
            data, 
            linkage_temp_clus['temp_clus'], 
            method=class_method, 
            depth=depth, 
            ntrees=ntrees,
            K_knn=K_knn, 
            dm0=dm0
        )
        
        # Final clustering assignment for centroid
        centroid_cluster_result = DAobs(
            data, 
            centroid_temp_clus['temp_clus'], 
            method=class_method, 
            depth=depth, 
            ntrees=ntrees,
            K_knn=K_knn, 
            dm0=dm0
        )
        
        return {
            'linkage_res': {
                'temp_clus': linkage_temp_clus['temp_clus'],
                'cluster_result': linkage_cluster_result,
                'group_list': linkage_temp_clus['group_list'],
                'stable_centers': linkage_temp_clus['stable_centers'],
                'strategy': 'linkage'
            },
            'centroid_res': {
                'temp_clus': centroid_temp_clus['temp_clus'],
                'cluster_result': centroid_cluster_result,
                'group_list': centroid_temp_clus['group_list'],
                'stable_centers': centroid_temp_clus['stable_centers'],
                'strategy': 'centroid'
            },
            'strategy': 'both'
        }
    else:
        # Handle single strategy (linkage or centroid)
        group_info = stable_centers_info['group_list']
        
        # Get temporary clustering
        temp_clus = get_temp_clus(
            sym_dm, 
            group_info, 
            strategy=stable_centers_info['strategy'], 
            data=data, 
            freq_table=freq_table
        )
        
        # Final clustering assignment
        cluster_result = DAobs(
            data, 
            temp_clus['temp_clus'], 
            method=class_method, 
            depth=depth, 
            ntrees=ntrees,
            K_knn=K_knn, 
            dm0=dm0
        )
        
        return {
            'temp_clus': temp_clus['temp_clus'],
            'cluster_result': cluster_result,
            'group_list': temp_clus['group_list'],
            'stable_centers': temp_clus['stable_centers'],
            'strategy': stable_centers_info['strategy']
        }

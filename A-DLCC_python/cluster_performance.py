import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import mode

def cluster_performance(y_true, y_pred):
    """
    Compute ARI, NMI, and purity for clustering result.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted cluster labels.
        
    Returns
    -------
    dict
        Dictionary with keys 'ARI', 'NMI', 'Purity'
    """
    # ARI
    ari = adjusted_rand_score(y_true, y_pred)
    # NMI
    nmi = normalized_mutual_info_score(y_true, y_pred)
    # Purity
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    clusters = np.unique(y_pred)
    correct = 0
    for c in clusters:
        mask = (y_pred == c)
        if np.sum(mask) == 0:
            continue
        correct += np.sum(y_true[mask] == mode(y_true[mask], keepdims=True).mode[0])
    purity = correct / len(y_true)
    
    return {'ARI': ari, 'NMI': nmi, 'Purity': purity}
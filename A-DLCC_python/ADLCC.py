import numpy as np
from typing import Dict, Optional, List
import spatial_dp as sd
from int_ld import integrated_ld
from auto_dlcc_functions import AUTO_DLCC, cluster2cv
from cluster_performance import cluster_performance


def ADLCC_wrapper(data: np.ndarray, 
                  labels: Optional[np.ndarray] = None,
                  class_method: str = 'knn',
                  strategy: Optional[List[str]] = None,
                  K_knn: int = 7,
                  depth: str = 'spatial',
                  ntrees: int = 100,
                  use_gpu: bool = False) -> Dict:
    """
    Complete ADLCC wrapper function that handles data size detection, 
    ILD calculation, AUTO_DLCC execution, and performance evaluation.
    
    Args:
        data: (n, d) data matrix
        labels: (n,) ground truth labels (optional, for performance evaluation)
        class_method: Classification method ('knn', 'maxdep', 'rf')
        strategy: Strategy to use ('linkage', 'centroid', 'both') or None for auto-selection
        K_knn: Number of neighbors for KNN
        depth: Depth method ('spatial')
        ntrees: Number of trees for random forest
        use_gpu: Whether to use GPU acceleration for spatial depth computation
        
    Returns:
        Dictionary containing:
            - dm_matrix: Output from spatial depth calculation (always included)
            - ILD_info: Integrated local depth information (always included)
            - AUTO_DLCC_output: Results from AUTO_DLCC (only if successful)
            - temp_performance: Performance metrics for temporary clustering (if labels provided and AUTO_DLCC successful)
            - final_performance: Performance metrics for final clustering (if labels provided and AUTO_DLCC successful)
            - temp_coverage: Percentage of data points in temporary clusters (if AUTO_DLCC successful)
            - final_coverage: Percentage of data points in final clusters (if AUTO_DLCC successful)
            
        Note: If AUTO_DLCC fails, only dm_matrix and ILD_info will be returned.
    """
    
    # Choose spatial depth method based on use_gpu parameter
    if use_gpu:
        # rspatial_dp_torch will automatically fallback to CPU if GPU memory insufficient
        rspatial_output = sd.rspatial_dp_torch(data, device='cuda')
    else:
        print(f"Using CPU for spatial depth computation")
        rspatial_output = sd.rspatial_dp(data)
    
    # Extract depth matrix
    dm0 = rspatial_output['dm']
    
    # Calculate Integrated Local Depth (ILD)
    ILD_info = integrated_ld(data, dm0, Lmatrix=rspatial_output['Lmatrix'])
    
    # Initialize result dictionary with first two items
    result = {
        'dm_matrix': rspatial_output,
        'ILD_info': ILD_info
    }
    
    # Try to run AUTO_DLCC
    try:
        AUTO_DLCC_output = AUTO_DLCC(
            ILD_info=ILD_info,
            dm0=dm0,
            data=data,
            class_method=class_method,
            K_knn=K_knn,
            depth=depth,
            strategy=strategy,
            ntrees=ntrees
        )
        
        # Add AUTO_DLCC output to result if successful
        result['AUTO_DLCC_output'] = AUTO_DLCC_output
        
    except Exception as e:
        print(f"AUTO_DLCC failed with error: {e}")
        print("Returning partial results (dm_matrix and ILD_info only)")
        return result
    
    # Calculate performance metrics if labels are provided and AUTO_DLCC was successful
    if labels is not None and 'AUTO_DLCC_output' in result:
        # Handle both strategy case
        if AUTO_DLCC_output['strategy'] == 'both':
            # Temporary clustering performance for both strategies
            linkage_temp_cv = cluster2cv(data, AUTO_DLCC_output['linkage_res']['temp_clus'])
            centroid_temp_cv = cluster2cv(data, AUTO_DLCC_output['centroid_res']['temp_clus'])
            
            # Final clustering performance for both strategies
            linkage_final_cv = AUTO_DLCC_output['linkage_res']['cluster_result']['cluster_vector']
            centroid_final_cv = AUTO_DLCC_output['centroid_res']['cluster_result']['cluster_vector']
            
            # Calculate coverage percentages
            linkage_temp_coverage = np.sum(linkage_temp_cv != 0) / len(linkage_temp_cv) * 100
            centroid_temp_coverage = np.sum(centroid_temp_cv != 0) / len(centroid_temp_cv) * 100
            linkage_final_coverage = np.sum(linkage_final_cv != 0) / len(linkage_final_cv) * 100
            centroid_final_coverage = np.sum(centroid_final_cv != 0) / len(centroid_final_cv) * 100
            
            # Calculate performance metrics for non-zero assignments
            linkage_temp_perf = cluster_performance(
                labels[linkage_temp_cv != 0], 
                linkage_temp_cv[linkage_temp_cv != 0]
            )
            centroid_temp_perf = cluster_performance(
                labels[centroid_temp_cv != 0], 
                centroid_temp_cv[centroid_temp_cv != 0]
            )
            linkage_final_perf = cluster_performance(
                labels[linkage_final_cv != 0], 
                linkage_final_cv[linkage_final_cv != 0]
            )
            centroid_final_perf = cluster_performance(
                labels[centroid_final_cv != 0], 
                centroid_final_cv[centroid_final_cv != 0]
            )
            
            result.update({
                'linkage_temp_performance': linkage_temp_perf,
                'centroid_temp_performance': centroid_temp_perf,
                'linkage_final_performance': linkage_final_perf,
                'centroid_final_performance': centroid_final_perf,
                'linkage_temp_coverage': linkage_temp_coverage,
                'centroid_temp_coverage': centroid_temp_coverage,
                'linkage_final_coverage': linkage_final_coverage,
                'centroid_final_coverage': centroid_final_coverage
            })
            
        else:
            # Single strategy case
            temp_cv = cluster2cv(data, AUTO_DLCC_output['temp_clus'])
            final_cv = AUTO_DLCC_output['cluster_result']['cluster_vector']
            
            # Calculate coverage percentages
            temp_coverage = np.sum(temp_cv != 0) / len(temp_cv) * 100
            final_coverage = np.sum(final_cv != 0) / len(final_cv) * 100
            
            # Calculate performance metrics for non-zero assignments
            temp_performance = cluster_performance(
                labels[temp_cv != 0], 
                temp_cv[temp_cv != 0]
            )
            final_performance = cluster_performance(
                labels[final_cv != 0], 
                final_cv[final_cv != 0]
            )
            
            result.update({
                'temp_performance': temp_performance,
                'final_performance': final_performance,
                'temp_coverage': temp_coverage,
                'final_coverage': final_coverage
            })
    
    return result


def print_performance_summary(result: Dict, strategy: Optional[str] = None):
    """
    Print a summary of clustering performance results.
    
    Args:
        result: Output from ADLCC_wrapper
        strategy: Strategy used ('linkage', 'centroid', 'both')
    """
    print(f"\n=== ADLCC Performance Summary ===")
    print(f"Strategy: {result['AUTO_DLCC_output']['strategy']}")
    
    if 'temp_performance' in result:
        # Single strategy case
        print(f"\nTemporary Clustering:")
        print(f"  Coverage: {result['temp_coverage']:.2f}%")
        print(f"  ARI: {result['temp_performance']['ARI']:.4f}")
        print(f"  NMI: {result['temp_performance']['NMI']:.4f}")
        print(f"  Purity: {result['temp_performance']['Purity']:.4f}")
        
        print(f"\nFinal Clustering:")
        print(f"  Coverage: {result['final_coverage']:.2f}%")
        print(f"  ARI: {result['final_performance']['ARI']:.4f}")
        print(f"  NMI: {result['final_performance']['NMI']:.4f}")
        print(f"  Purity: {result['final_performance']['Purity']:.4f}")
        
    elif 'linkage_temp_performance' in result:
        # Both strategy case
        print(f"\nLinkage Strategy:")
        print(f"  Temporary - Coverage: {result['linkage_temp_coverage']:.2f}%, "
              f"ARI: {result['linkage_temp_performance']['ARI']:.4f}, "
              f"NMI: {result['linkage_temp_performance']['NMI']:.4f}, "
              f"Purity: {result['linkage_temp_performance']['Purity']:.4f}")
        print(f"  Final - Coverage: {result['linkage_final_coverage']:.2f}%, "
              f"ARI: {result['linkage_final_performance']['ARI']:.4f}, "
              f"NMI: {result['linkage_final_performance']['NMI']:.4f}, "
              f"Purity: {result['linkage_final_performance']['Purity']:.4f}")
        
        print(f"\nCentroid Strategy:")
        print(f"  Temporary - Coverage: {result['centroid_temp_coverage']:.2f}%, "
              f"ARI: {result['centroid_temp_performance']['ARI']:.4f}, "
              f"NMI: {result['centroid_temp_performance']['NMI']:.4f}, "
              f"Purity: {result['centroid_temp_performance']['Purity']:.4f}")
        print(f"  Final - Coverage: {result['centroid_final_coverage']:.2f}%, "
              f"ARI: {result['centroid_final_performance']['ARI']:.4f}, "
              f"NMI: {result['centroid_final_performance']['NMI']:.4f}, "
              f"Purity: {result['centroid_final_performance']['Purity']:.4f}")
    
    else:
        print("No performance metrics available (labels not provided)")

import numpy as np

def spatial_depth(x, data, Lmatrix=None):
    data = np.asarray(data)
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]   # (1, d)
    n, d = data.shape
    N = x.shape[0]


    if Lmatrix is None:
        dep = np.zeros(N)
        t_matrix = np.eye(d)
        for i in range(N):
            a = np.matmul(t_matrix, (x[i][:, None] - data.T))  # d × n
            a = a.T                                            # n × d
            abs_row_sum = np.sum(np.abs(a), axis=1)
            a = a[abs_row_sum != 0, :]                         
            norms = np.sqrt(np.sum(a**2, axis=1))              # Euclidean norm
            a = a / norms[:, None]
            e = np.sum(a, axis=0) / n
            dep[i] = 1 - np.sqrt(np.sum(e**2))
        return dep
    else:
        tol = 1e-5
        Lmatrix = np.copy(Lmatrix)
        Lmatrix[Lmatrix < tol] = np.inf
        Lmatrix = 1 / Lmatrix
        C = data.T @ Lmatrix      # d × n
        C = C.T                   # n × d
        norm_csum = np.sum(Lmatrix, axis=0)    # shape: n
        C2 = norm_csum[:, None] * data         # n × d
        C = C2 - C                             # n × d
        dep = 1 - np.sqrt(np.sum((C / n) ** 2, axis=1))
        return dep

def rspatial_dp(data):
    data = np.asarray(data)
    n, d = data.shape
    rn_inv = 1 / (2 * n - 1)
    dm = np.zeros((n, n))
    tol = 1e-5
    Ematrix = np.zeros((n, d))
    Lmatrix = np.zeros((n, n))
    for i in range(n):
        a = -data + data[i]
        Lmatrix[:, i] = np.linalg.norm(a, axis=1)
        norm_a = Lmatrix[:, i].copy()
        norm_a[norm_a < tol] = 1
        a = a / norm_a[:, None]
        Ematrix[i] = np.sum(a, axis=0)
    Lmatrix_save = Lmatrix.copy()
    Lmatrix = Lmatrix ** 2

    two_Lmatrix = 2 * Lmatrix
    for j in range(n):
        idx = np.arange(n) != j
        b_temp = data[idx] + (-2 * data[j])
        b1 = two_Lmatrix[idx, j][:, None]
        b2 = two_Lmatrix[j, :]
        lsub = Lmatrix[idx, :]
        norm_bM = np.sqrt(np.abs(b1 + b2 - lsub))
        norm_bM[norm_bM < tol] = np.inf
        norm_bM = 1 / norm_bM
        C = (b_temp.T @ norm_bM).T  # 矩阵乘法
        norm_csum = np.sum(norm_bM, axis=0)
        C2 = norm_csum[:, None] * data
        C = C + C2 + Ematrix
        dm[j] = 1 - np.linalg.norm(C * rn_inv, axis=1)
    return {"dm": dm, "Lmatrix": Lmatrix_save}


def rspatial_dp_torch(data, device='cuda'):
    """
    PyTorch implementation with GPU memory estimation and smart device selection
    """
    import torch
    
    data = np.asarray(data)
    n, d = data.shape
    
    # Estimate memory requirements (in bytes)
    # Main tensors: data(n×d), dm(n×n), Ematrix(n×d), Lmatrix(n×n), Lmatrix_save(n×n)
    # Plus temporary variables in loops (estimated as additional n×n)
    memory_needed = (2 * n * d + 5 * n * n) * 8  # float64 = 8 bytes
    memory_needed_mb = memory_needed / (1024 * 1024)
    
    # Check if GPU is available and has enough memory
    if device == 'cuda' and torch.cuda.is_available():
        try:
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            gpu_free_mb = gpu_memory_mb - (torch.cuda.memory_allocated() / (1024 * 1024))
            
            # Use 80% of free memory as safety margin
            if memory_needed_mb < gpu_free_mb * 0.6:
                print(f"Estimated memory: {memory_needed_mb:.1f}MB, GPU available: {gpu_free_mb:.1f}MB - Using GPU")
                device = 'cuda'
            else:
                print(f"Estimated memory: {memory_needed_mb:.1f}MB exceeds available GPU memory: {gpu_free_mb:.1f}MB - Using CPU")
                device = 'cpu'
        except:
            print("Could not check GPU memory, using CPU")
            device = 'cpu'
    else:
        device = 'cpu'
        print("Using CPU for spatial depth computation")
    
    try:
        data = torch.as_tensor(data, dtype=torch.float64, device=device)
        
        rn_inv = 1.0 / (2 * n - 1)
        dm = torch.zeros((n, n), dtype=torch.float64, device=device)
        tol = 1e-5
        Ematrix = torch.zeros((n, d), dtype=torch.float64, device=device)
        Lmatrix = torch.zeros((n, n), dtype=torch.float64, device=device)

        for i in range(n):
            a = -data + data[i]
            Lmatrix[:, i] = torch.norm(a, dim=1)
            norm_a = Lmatrix[:, i].clone()
            norm_a[norm_a < tol] = 1
            a = a / norm_a[:, None]
            Ematrix[i] = a.sum(dim=0)
        Lmatrix_save = Lmatrix.clone()
        Lmatrix = Lmatrix ** 2

        for j in range(n):
            idx = torch.arange(n, device=device) != j
            b_temp = data[idx] + (-2 * data[j])
            b1 = (2 * Lmatrix[idx, j]).unsqueeze(1)
            b2 = (2 * Lmatrix[j, :]).unsqueeze(0)
            lsub = Lmatrix[idx, :]
            norm_bM = torch.sqrt(torch.abs(b1 + b2 - lsub))
            norm_bM[norm_bM < tol] = float('inf')
            norm_bM = 1 / norm_bM
            C = (b_temp.T @ norm_bM).T
            norm_csum = norm_bM.sum(dim=0)
            C2 = norm_csum.unsqueeze(1) * data
            C = C + C2 + Ematrix
            dm[j] = 1 - torch.norm(C * rn_inv, dim=1)
            
        return {"dm": dm.cpu().numpy(), "Lmatrix": Lmatrix_save.cpu().numpy()}
        
    except Exception as e:
        # Fallback to numpy implementation if torch fails
        print(f"PyTorch computation failed ({e}), falling back to numpy...")
        return rspatial_dp(data.cpu().numpy() if hasattr(data, 'cpu') else data)
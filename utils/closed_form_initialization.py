import torch


@torch.no_grad()
def intial_value(X1_list, X2_list, W1, W2, idx_list,
                     lam=1, iters_refine=1, dtype_acc=torch.float64):
    
    
    device = W1.device
    d1, k  = W1.shape
    

    # 누적은 고정밀
    Sxx = torch.zeros((d1, d1), dtype=dtype_acc, device=device)
    Sxy = torch.zeros((d1, k),   dtype=dtype_acc, device=device)

    for i in idx_list:
        X1 = X1_list[i].to(device, dtype=dtype_acc)      
        X2 = X2_list[i].to(device, dtype=dtype_acc)         
        Y  = X1.to(dtype_acc)[:,:d1] @ W1.to(dtype_acc) + X2 @ W2.to(dtype_acc)   


        Sxx = torch.addmm(Sxx, X1.transpose(0,1), X1, beta=1.0, alpha=1.0)
        Sxy = torch.addmm(Sxy, X1.transpose(0,1), Y,  beta=1.0, alpha=1.0)

    if lam is None or lam <= 0:
        lam_eff = 0.0
    else:
        dmean = torch.clamp(torch.mean(torch.diag(Sxx)), min=1e-18)
        lam_eff = float(lam * dmean)


    Sxx = 0.5 * (Sxx + Sxx.transpose(0,1))
    Sxx.diagonal().add_(lam_eff)

    jitter = 0.0
    for _ in range(3):
        try:
            L = torch.linalg.cholesky(Sxx)
            W3 = torch.cholesky_solve(Sxy, L)                 
            break
        except RuntimeError:
            jitter = max(1e-18, 10.0 * (jitter if jitter > 0 else lam_eff or 1e-18))
            Sxx.diagonal().add_(jitter)
    else:
        W3 = torch.linalg.pinv(Sxx) @ Sxy

    
    for _ in range(iters_refine):
        r = Sxy - Sxx @ W3
        try:
            L = torch.linalg.cholesky(Sxx)
            delta = torch.cholesky_solve(r, L)
        except RuntimeError:
            delta = torch.linalg.pinv(Sxx) @ r
        W3 = W3 + delta

    return W3.to(W1.dtype)

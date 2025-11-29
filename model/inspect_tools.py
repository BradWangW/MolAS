import torch
import numpy as np

def random_so3(pos, device):
    q = torch.randn(4, device=device)
    q = q / q.norm()
    w, x, y, z = q
    R = torch.tensor([
      [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
      [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
      [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], device=device)
    pos = pos @ R.t()
    return pos

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def linear_CKA(X, Y, eps=1e-8):
    # X,Y: [N, D] zero-centered
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.t()
    L = Y @ Y.t()
    hsic = (K * L).sum()
    nX = torch.linalg.norm(K)
    nY = torch.linalg.norm(L)
    return (hsic / (nX*nY + eps)).item()

def mean_pairwise_cosine(X):
    # X: [N, D] node embeddings for one graph
    X = F.normalize(X, dim=-1)
    S = X @ X.t()
    n = S.size(0)
    return (S.sum() - n) / (n*(n-1) + 1e-8)

@torch.no_grad()
def oversmoothing_curve(layer_outputs): 
    # layer_outputs: list of [N,D] (per layer) for the SAME batch/graph
    return [mean_pairwise_cosine(H).item() for H in layer_outputs]

def reliability_curve(scores, labels, n_bins=10):
    scores = np.asarray(scores); labels = np.asarray(labels).astype(int)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (scores >= lo) & (scores < hi) if i < n_bins-1 else (scores >= lo) & (scores <= hi)
        if idx.sum() == 0:
            bin_stats.append(( (lo+hi)/2, np.nan, np.nan, 0 ))
            continue
        conf = scores[idx].mean()
        acc  = labels[idx].mean()
        ece += (idx.mean()) * abs(acc - conf)
        bin_stats.append(((lo+hi)/2, conf, acc, idx.sum()))
    return ece, bin_stats
# weighted network generator
import numpy as np
import torch


def k_s_generator(N, connectance=0.5, max_w=1000):
    # get the topology
    matrix = np.random.uniform(0, 1, (N, N))
    matrix-=np.diag(matrix.diagonal())
    matrix[matrix< 1-connectance] = 0  
    matrix = torch.tensor(matrix, dtype=torch.float32)
    A = torch.where(matrix > 0, 1, 0)  # Binarizza la matrice
    W = np.random.randint(1, max_w, (N, N))
    W = torch.tensor(W, dtype=torch.float32)
    W=(W*A)
    assert torch.all(torch.where(W > 0, 1, 0)==A)
    s=torch.cat((W.sum(1), W.sum(0)))
    k=torch.cat((A.sum(1), A.sum(0)))
    return k, s

def k_s_generator_pl(N, rho=1e-3, seed=None, alpha_pareto = 2.5):
    """Generate test network, using a CL model for the topology, w/ power law distributed degrees and strengths"""
    np.random.seed(seed)
    #torch.manual_seed(seed)
    
    # Topology
    x_out = np.random.pareto(alpha_pareto, size=N) + 1.0
    x_in = np.random.pareto(alpha_pareto, size=N) + 1.0
    
    expected_L=np.round(N*(N-1)*rho).astype(int)
    scale_out = np.sqrt(expected_L) / x_out.sum()
    scale_in = np.sqrt(expected_L) / x_in.sum()
    x_out *= scale_out
    x_in *= scale_in
    
    k=np.zeros(2*N)
    s=np.zeros(2*N)
    
    for i in range(N):
        # First, topology...
        row_i=np.random.random(size=N)<=x_out[i]*x_in
        # adjusting the diagonal
        row_i[i]=0
        # adjusting Chung-Lu weird entries
        where_g1=np.where(x_out[i]*x_in>1)[0]
        if len(where_g1)>0:
            for j in where_g1:
                row_i[j]=1
        
        k[i]=row_i.sum()
        k[N:]+=row_i
        
        # ...then weights
        where_g0=np.where(row_i>0)[0]
        w_is=np.random.pareto(alpha_pareto, size=len(where_g0)) + 1.0
        w_is=np.ceil(w_is)
        s[i]=w_is.sum()
        for ij, j in enumerate(where_g0):
            s[N+j]+=w_is[ij]

    k = torch.from_numpy(k).int()
    s = torch.from_numpy(s).int()
    return k, s

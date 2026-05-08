import torch
from torch_scatter import scatter_mean


def embed_T(T, s_idx, device, T_norm=5000.0):
       
    T = torch.as_tensor(T).reshape(1,1).to(device)
    e_T = torch.cat([torch.ones_like(T), T/T_norm, (T/T_norm)**2, (T/T_norm)**3],-1).repeat(len(s_idx),1)
 
    return e_T

def embed_x(node_attr, edge_attr, edge_index, s_idx):
    e_k=scatter_mean(torch.cat([node_attr[edge_index[0]], node_attr[edge_index[1]], edge_attr],-1), edge_index[0], 0, dim_size=len(node_attr))[s_idx]
    e_x=torch.cat([e_k, e_k**2, e_k**3],-1)
    return e_x





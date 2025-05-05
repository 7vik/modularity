import torch

def clusterability(matrix, auto_index=True, cluster_U_indices=None, cluster_V_indices=None, num_clusters=4):

    if auto_index:
        cluster_size = (matrix.shape[0] // num_clusters, matrix.shape[1] // num_clusters)
        cluster_U_indices = {i: list(range(i*cluster_size[0], (i+1)*cluster_size[0])) for i in range(num_clusters)}
        cluster_V_indices = {i: list(range(i*cluster_size[1], (i+1)*cluster_size[1])) for i in range(num_clusters)}

    num_clusters = len(cluster_U_indices)
    A = matrix ** 2
    mask = torch.zeros_like(A, dtype=torch.bool)
    
    for cluster_idx in range(num_clusters):
        u_indices = torch.tensor(cluster_U_indices[cluster_idx], dtype=torch.long)
        v_indices = torch.tensor(cluster_V_indices[cluster_idx], dtype=torch.long)
        mask[u_indices.unsqueeze(1), v_indices] = True
    
    intra_cluster_out_sum = torch.sum(A[mask])
    total_out_sum = torch.sum(A)
    
    return intra_cluster_out_sum / total_out_sum
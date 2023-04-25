import copy
import torch
import numpy as np


def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def DaAgg(w, dict_len, clean_clients, noisy_clients):
    client_weight = np.array(dict_len)
    client_weight = client_weight / client_weight.sum()
    distance = np.zeros(len(dict_len))
    for n_idx in noisy_clients:
        dis = []
        for c_idx in clean_clients:
            dis.append(model_dist(w[n_idx], w[c_idx]))
        distance[n_idx] = min(dis)
    distance = distance / distance.max()
    client_weight = client_weight * np.exp(-distance)
    client_weight = client_weight / client_weight.sum()
    # print(client_weight)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * client_weight[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * client_weight[i]
    return w_avg


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1.keys():
        if "int" in str(w_1[key].dtype):
            continue
        dist = torch.norm(w_1[key] - w_2[key])
        dist_total += dist.cpu()

    return dist_total.cpu().item()
import torch
import torch.nn.functional as F
import numpy as np

def find_edge(true_label, neighbor_num=4):
    '''
    :param true_label: shape-[BS, H, W], 1 if pixels belong to foreground, otherwise 0
    :return: edge_flag, shape-[BS, H, W]
    '''
    H = true_label.shape[-2]
    W = true_label.shape[-1]

    # The edge pixels possess value bigger than 0
    if neighbor_num == 4:
        edge_flag = torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 0:H - 2, 1:W - 1]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 2:H, 1:W - 1]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 1:H - 1, 0:W - 2]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 1:H - 1, 2:W])
    elif neighbor_num == 8:
        edge_flag = torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 0:H - 2, 1:W - 1]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 2:H, 1:W - 1]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 1:H - 1, 0:W - 2]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 1:H - 1, 2:W]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 0:H - 2, 0:W - 2]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 2:H, 0:W - 2]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 0:H - 2, 2:W]) + \
                    torch.abs(true_label[:, 1:H - 1, 1:W - 1] - true_label[:, 2:H, 2:W])
    else:
        edge_flag = None
    edge_flag = (edge_flag > 0).long()

    return edge_flag

def find_edge_and_neighbor(true_label, neighbor_num=4):
    '''
    :param true_label: shape-[BS, H, W], 1 if pixels belong to foreground, otherwise 0
    :return: (foreground_edge_flag, background_edge_flag, foreground_edge_neighbor_flag), shape-[BS, H, W]
    '''
    H = true_label.shape[-2]
    W = true_label.shape[-1]
    edge_flag_dummy = torch.zeros(true_label.shape).cuda()
    fg_edge_neighbor_flag_dummy = torch.zeros(true_label.shape).cuda()
    bg_edge_neighbor_flag_dummy = torch.zeros(true_label.shape).cuda()

    # Getting foreground & background edge pixels
    edge_flag = find_edge(true_label, neighbor_num=4)
    edge_flag_dummy[:, 1:H - 1, 1:W - 1] = edge_flag  # Deal with reduction of H W
    foreground_edge_flag = true_label * edge_flag_dummy
    background_edge_flag = (1-true_label) * edge_flag_dummy

    # Getting foreground edge neighbor pixels
    fg_edge_neighbor_flag = find_edge(foreground_edge_flag, neighbor_num=8)
    fg_edge_neighbor_flag_dummy[:, 1:H - 1, 1:W - 1] = fg_edge_neighbor_flag  # Deal with reduction of H W
    foreground_edge_neighbor_flag = fg_edge_neighbor_flag_dummy * true_label * (1-foreground_edge_flag)

    # Getting background edge neighbor pixels
    bg_edge_neighbor_flag = find_edge(background_edge_flag, neighbor_num=8)
    bg_edge_neighbor_flag_dummy[:, 1:H - 1, 1:W - 1] = bg_edge_neighbor_flag  # Deal with reduction of H W
    background_edge_neighbor_flag = bg_edge_neighbor_flag_dummy * (1-true_label) * (1-background_edge_flag)

    return foreground_edge_flag, foreground_edge_neighbor_flag, background_edge_flag, background_edge_neighbor_flag

def sort_edge(edge, pred_probs):
    '''
    :param edge: shape-[BS, H, W], result from Function::find_edge
    :param pred_probs: shape-[BS, 1, H, W], the probability of foreground class
    :return: sort_value, sort_indices, num_of_edge_pixels
    '''

    BS = edge.shape[0]

    # Flatten and get edge probs
    flat_edge_flag = edge.view(BS, -1)
    flat_probs = pred_probs.squeeze(1).view(BS, -1)
    flat_edge_probs = flat_edge_flag * flat_probs  # flat_edge_flag as mask, non-edge pixel will be zero

    # Sort and Slice edge cases
    num_of_edge_pixels = flat_edge_flag.sum(1)  # [BS], index where the last edge case is
    sort_value, sort_indices = torch.sort(flat_edge_probs, dim=1, descending=True)

    return sort_value, sort_indices, num_of_edge_pixels

def get_slice_index(anchor_point, k=10, backward=True):
    result = []
    for i in range(anchor_point.shape[0]):
        anchor_point_temp = anchor_point[i].long().item()
        if backward:
            slice_temp = torch.tensor(range(anchor_point_temp-k+1, anchor_point_temp))
        else:
            slice_temp = torch.tensor(range(anchor_point_temp, anchor_point_temp+k))
        result.append(slice_temp)
    result = torch.stack(result, dim=0).cuda()
    return result

def get_triplet(true_label, pred_probs, embeddings, n=10, m=10):
    '''
    :param true_label: shape-[BS, H, W], 1 if pixels belong to foreground, otherwise 0
    :param pred_probs: shape-[BS, 1, H, W], the probability of foreground class
    :param embeddings: shape-[BS, emb_size, H, W]
    :param n: number of anchor
    :param m: number of positive and negative samples
    :return: anchor_result, p1_result, p2_result, p1_probs, n1_result, n2_result, n1_probs
    '''
    BS = true_label.shape[0]
    emb_dim = embeddings.shape[1]
    embs = embeddings.view(BS, emb_dim, -1)  # [BS, emb_size, HxW]
    embs = embs.permute(0, 2, 1)  # [BS, HxW, emb_size]

    # Get edge & edge neighbor for foreground & background edge
    foreground_edge_flag, fg_edge_neighbor_flag, background_edge_flag, bg_edge_neighbor_flag = find_edge_and_neighbor(true_label)


    # #####################################################
    # Part 1: Foreground (fg), neighbor (nb)
    fg_sort_value, fg_sort_indices, fg_num_of_edge_pixels = sort_edge(foreground_edge_flag, pred_probs)
    fg_nb_sort_value, fg_nb_sort_indices, fg_nb_num_of_edge_pixels = sort_edge(fg_edge_neighbor_flag, pred_probs)

    # Anchor
    anchor_slice_index = get_slice_index(fg_num_of_edge_pixels, k=n, backward=True)

    # In case there is negative index
    anchor_slice_index_positive = (anchor_slice_index >= 0)
    anchor_slice_index *= anchor_slice_index_positive
    valid_anchor_flag = (anchor_slice_index.sum(-1) > 0).detach().cpu().numpy()

    anchor_index = torch.gather(fg_sort_indices, 1, anchor_slice_index)
    anchor_result = []
    for i in range(BS):
        anchor_temp = torch.index_select(embs[i], 0, anchor_index[i])
        anchor_result.append(anchor_temp)
    anchor_result = torch.stack(anchor_result, dim=0)

    # Positive_Set (first)
    p1_slice_index = get_slice_index(torch.zeros(fg_num_of_edge_pixels.shape), k=m, backward=False)  # get the first k pixels that predict well
    p1_index = torch.gather(fg_sort_indices, 1, p1_slice_index)
    p1_probs = torch.gather(fg_sort_value, 1, p1_slice_index).mean(-1)  # P_S_p1, the probablity of choosing S_p1
    p1_result = []
    for i in range(BS):
        p1_temp = torch.index_select(embs[i], 0, p1_index[i])
        p1_result.append(p1_temp)
    p1_result = torch.stack(p1_result, dim=0)

    # Positive_Set (second)
    p2_slice_index = get_slice_index(torch.zeros(fg_nb_num_of_edge_pixels.shape), k=m, backward=False)
    p2_index = torch.gather(fg_nb_sort_indices, 1, p2_slice_index)
    p2_result = []
    for i in range(BS):
        p2_temp = torch.index_select(embs[i], 0, p2_index[i])
        p2_result.append(p2_temp)
    p2_result = torch.stack(p2_result, dim=0)


    # #####################################################
    # Part 2: Background (bg)
    bg_sort_value, bg_sort_indices, bg_num_of_edge_pixels = sort_edge(background_edge_flag, (1-pred_probs))
    bg_nb_sort_value, bg_nb_sort_indices, bg_nb_num_of_edge_pixels = sort_edge(bg_edge_neighbor_flag, (1-pred_probs))

    # Negative_Set (first)
    n1_slice_index = get_slice_index(torch.zeros(bg_num_of_edge_pixels.shape), k=m, backward=False)
    n1_index = torch.gather(bg_sort_indices, 1, n1_slice_index)
    n1_probs = torch.gather(bg_sort_value, 1, n1_slice_index).mean(-1)  # P_S_n1, the probablity of choosing S_n1
    n1_result = []
    for i in range(BS):
        n1_temp = torch.index_select(embs[i], 0, n1_index[i])
        n1_result.append(n1_temp)
    n1_result = torch.stack(n1_result, dim=0)

    # Negative_Set (second)
    n2_slice_index = get_slice_index(torch.zeros(bg_nb_num_of_edge_pixels.shape), k=m, backward=False)
    n2_index = torch.gather(bg_nb_sort_indices, 1, n2_slice_index)
    n2_result = []
    for i in range(BS):
        n2_temp = torch.index_select(embs[i], 0, n2_index[i])
        n2_result.append(n2_temp)
    n2_result = torch.stack(n2_result, dim=0)


    return anchor_result, p1_result, p2_result, p1_probs, n1_result, n2_result, n1_probs, valid_anchor_flag


def cal_triplet_loss(anchor_result, p1_result, p2_result, p1_probs, n1_result, n2_result, n1_probs, loss_func_triplet, valid_anchor_flag):
    BS = anchor_result.shape[0]  # batch size
    AN = anchor_result.shape[1]  # anchor number
    PNN = p1_result.shape[1]  # positive & negative sample number

    triplet_loss_result = []
    for bs in range(BS):
        for an, valid_flag in zip(range(AN), valid_anchor_flag):
            if not valid_flag:
                continue
            anchor_temp = anchor_result[bs][an]
            p1_probs_temp = p1_probs[bs].item()
            positive = p1_result[bs][np.random.randint(PNN)] if np.random.randint(100) < p1_probs_temp * 100 else p2_result[bs][np.random.randint(PNN)]
            negative = n1_result[bs][np.random.randint(PNN)] if np.random.randint(100) < p1_probs_temp * 100 else n2_result[bs][np.random.randint(PNN)]

            triplet_loss_result.append(loss_func_triplet(anchor_temp.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0)))

    if len(triplet_loss_result) == 0:
        return torch.zeros(1).cuda()
    else:
        return torch.stack(triplet_loss_result, 0).mean()
    # batch_loss_triplet = loss_func_triplet(feature_anchor, feature_pos, feature_neg)

def get_triplet_loss(true_label, pred_probs, embeddings, loss_func_triplet, n=10, m=10):
    anchor_result, p1_result, p2_result, p1_probs, n1_result, n2_result, n1_probs, valid_anchor_flag = get_triplet(true_label, pred_probs, embeddings)
    batch_loss_triplet = cal_triplet_loss(anchor_result, p1_result, p2_result, p1_probs, n1_result, n2_result, n1_probs, loss_func_triplet, valid_anchor_flag)

    return batch_loss_triplet
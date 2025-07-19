# from Kelly_data_preprocess.ransac_icp_svd import eva_regist
from tqdm import tqdm
import numpy as np
import torch

def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0]).double()
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H.cpu())  # H = USV^T
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V.double() @ eye.double() @ Ut.double()

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t

def stat(x, y, bins):
    group_index = []
    results_mean = []
    results_std = []

    for i in np.arange(len(bins) - 1):
        idx = np.where(np.logical_and(x >= bins[i], x <= bins[i + 1]))[0]
        group_index.append(idx)
        sub_result_mean = np.mean(y[idx])
        sub_result_std = np.std(y[idx])
        results_std.append(sub_result_std)
        results_mean.append(sub_result_mean)

    return group_index, results_mean, results_std


if __name__ == '__main__':


    test_root_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/in_vitro/Rigid_test_data/"
    test_list_file_path = test_root_path + "rigid_list.npy"
    save_stat_file_path = test_root_path + "stat.npz"
    test_list = np.load(test_list_file_path)

    vis = []
    deform = []

    for index in tqdm(range(len(test_list))):
        data_file = test_root_path + test_list[index]

        entry = np.load(data_file)
        src_vs = entry['src_vs']
        tgt_vs = entry['tgt_vs']
        # flow = entry['flow']

        src_markers = entry['src_marker']
        tgt_markers = entry['tgt_marker']

        R, t = weighted_procrustes(torch.from_numpy(src_markers).double().cuda(),
                                   torch.from_numpy(tgt_markers).double().cuda())
        R = R.cpu().numpy()
        t = t.cpu().numpy()[:, np.newaxis]
        src_marker_pred = (np.matmul(R, src_markers.T) + t).T

        flow = tgt_markers - src_marker_pred

        RE = np.sqrt(np.sum(flow * flow) / len(flow))
        vis_value = len(tgt_vs) / len(src_vs)
        deform.append(RE)
        vis.append(vis_value)

    center_vis = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bins_vis = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    vis = np.array(vis)
    deform = np.array(deform)

    _, mean_reg_v, std_reg_v = stat(vis, deform, bins_vis)

    print("vis")
    for i in np.arange(len(mean_reg_v)):
        print("&", round(mean_reg_v[i], 2), "$\pm$", round(std_reg_v[i], 2), "\n ")

    np.savez(save_stat_file_path, deform=deform, vis=vis)


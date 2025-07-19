import numpy as np
from tqdm import tqdm

# def weighted_procrustes(
#     src_points,
#     ref_points,
#     weights=None,
#     weight_thresh=0.0,
#     eps=1e-5,
#     return_transform=False,
# ):
#     r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.
#
#     Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).
#
#     Args:
#         src_points: torch.Tensor (B, N, 3) or (N, 3)
#         ref_points: torch.Tensor (B, N, 3) or (N, 3)
#         weights: torch.Tensor (B, N) or (N,) (default: None)
#         weight_thresh: float (default: 0.)
#         eps: float (default: 1e-5)
#         return_transform: bool (default: False)
#
#     Returns:
#         R: torch.Tensor (B, 3, 3) or (3, 3)
#         t: torch.Tensor (B, 3) or (3,)
#         transform: torch.Tensor (B, 4, 4) or (4, 4)
#     """
#     if src_points.ndim == 2:
#         src_points = src_points.unsqueeze(0)
#         ref_points = ref_points.unsqueeze(0)
#         if weights is not None:
#             weights = weights.unsqueeze(0)
#         squeeze_first = True
#     else:
#         squeeze_first = False
#
#     batch_size = src_points.shape[0]
#     if weights is None:
#         weights = torch.ones_like(src_points[:, :, 0]).double()
#     weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
#     weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
#     weights = weights.unsqueeze(2)  # (B, N, 1)
#
#     src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
#     ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
#     src_points_centered = src_points - src_centroid  # (B, N, 3)
#     ref_points_centered = ref_points - ref_centroid  # (B, N, 3)
#
#     H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
#     U, _, V = torch.svd(H.cpu())  # H = USV^T
#     Ut, V = U.transpose(1, 2).cuda(), V.cuda()
#     eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
#     eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
#     R = V @ eye @ Ut
#
#     t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
#     t = t.squeeze(2)
#
#     if return_transform:
#         transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
#         transform[:, :3, :3] = R
#         transform[:, :3, 3] = t
#         if squeeze_first:
#             transform = transform.squeeze(0)
#         return transform
#     else:
#         if squeeze_first:
#             R = R.squeeze(0)
#             t = t.squeeze(0)
#         return R, t

if __name__ == '__main__':

    root_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/"
    test_list = root_path + "Deform_mesh_npz_test/list.npz"
    test_path = root_path + "Deform_mesh_npz_test/Test/"
    test_list = np.load(test_list)['test']

    vis = []
    deform = []
    max_deform_list = []
    init_list = []

    for index in tqdm(range(len(test_list))):
        data_file = test_path + test_list[index]

        entry = np.load(data_file)

        src_vs = entry['src_pcd']
        tgt_vs = entry['tgt_pcd']
        # flow = entry['flow']

        src_markers = entry['src_vol']
        tgt_markers = entry['tgt_vol']

        rot_tgt = entry['rot_tgt']
        # src_markers = (np.matmul(rot_src, src_markers.T)).T
        tgt_markers_test = (np.matmul(rot_tgt, tgt_markers.T)).T

        flow_init = tgt_markers_test - src_markers

        RE_init = np.sqrt(np.sum(flow_init * flow_init) / len(flow_init))


        R = entry['R_gt']
        t = entry['t_gt']

        src_marker_pred = (np.matmul(R, src_markers.T) + t).T

        flow = tgt_markers - src_marker_pred

        max_deform = np.max(np.sqrt(np.sum(flow * flow, axis=1)))

        RE = np.sqrt(np.sum(flow * flow) / len(flow))
        vis_value = len(tgt_vs) / len(src_vs)

        deform.append(RE)
        max_deform_list.append(max_deform)
        vis.append(vis_value)
        init_list.append(RE_init)

    np.savez("/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/stat_svd.npz", init=init_list, deform=deform, max_deform=max_deform_list, vis=vis)


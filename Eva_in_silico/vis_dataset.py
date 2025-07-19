import numpy as np
import pyvista as pv
import open3d as o3d
import torch

def pad_faces(faces):
    """input faces: nx3"""
    n, _ = np.shape(faces)
    faces_ = np.ones([n, 4]) * 3
    faces_[:, 1:] = faces
    return np.hstack(faces_).astype(int)

def load_test_from_npz(file, sigma=None, rot=False):
    with np.load(file, allow_pickle=True) as entry:
        src_vs = entry['src_pcd']
        src_f = entry['src_f']
        src_f = pad_faces(src_f)
        #src_vox = entry['src_vox']
        # edges = entry['src_edges']
        tgt_vs = entry['tgt_pcd']
        tgt_f = entry['tgt_f']
        #tgt_vox = entry['tgt_vox']
        flow = entry['flow']
        rot_src = entry['rot_src']
        rot_tgt = entry['rot_tgt']
        tgt_full = src_vs + flow

        src_markers = entry['src_vol']
        tgt_markers = entry['tgt_vol']

        if sigma is not None:

            noise = entry[str(sigma)]
            tgt_vs = tgt_vs + noise

        if rot:

            src_vs = (np.matmul(rot_src, src_vs.T)).T
            tgt_vs = (np.matmul(rot_tgt, tgt_vs.T)).T

            # src_vox = (np.matmul(rot_src, src_vox.T)).T
            # tgt_vox = (np.matmul(rot_tgt, tgt_vox.T)).T

            src_markers = (np.matmul(rot_src, src_markers.T)).T
            tgt_markers = (np.matmul(rot_tgt, tgt_markers.T)).T

            tgt_full = (np.matmul(rot_tgt, tgt_full.T)).T
            flow = tgt_full - src_vs

    return src_vs, src_f, tgt_vs, tgt_f, tgt_full, flow, src_markers, tgt_markers #, src_vox, tgt_vox

def vis_mesh_pc(src_vs=None, src_faces=None, tgt_vs=None):
    plotter = pv.Plotter()
    plotter.set_background('white')

    surf = pv.PolyData(src_vs, src_faces)
    plotter.add_mesh(surf)
    if tgt_vs is not None:
        plotter.add_points(tgt_vs, point_size=10, render_points_as_spheres=True, color='red')
    # if src_vs is not None:
    #     plotter.add_points(src_vs, color='blue')
    plotter.show()


def compare_mesh(src_vs=None, src_faces=None, tgt_vs=None, tgt_faces=None, opacity=0.6):
    plotter = pv.Plotter()
    plotter.set_background('white')

    src_surf = pv.PolyData(src_vs, src_faces)
    plotter.add_mesh(src_surf, color='blue', show_edges=True, opacity=opacity)

    if tgt_vs is not None and tgt_faces is not None:

        tgt_surf = pv.PolyData(tgt_vs, tgt_faces)
        plotter.add_mesh(tgt_surf,  color='red', show_edges=True, opacity=opacity)

    plotter.show()


def vis_pc_marker(src_vs=None, src_marker=None, tgt_vs=None, tgt_marker=None, point_opacity=1, marker_opacity=0.6, point_size=12, marker_size=10):

    plotter = pv.Plotter()
    plotter.set_background('white')

    if src_vs is not None:
        plotter.add_points(src_vs, color=[0, 150, 255], point_size=point_size, render_points_as_spheres=True, opacity=point_opacity)

    if tgt_vs is not None:
        plotter.add_points(tgt_vs, color=[254, 92, 92], point_size=point_size, render_points_as_spheres=True, opacity=point_opacity)

    if src_marker is not None:
        plotter.add_points(src_marker, color=[175, 175, 175], point_size=marker_size, render_points_as_spheres=True, opacity=marker_opacity) #,render_points_as_spheres=True, point_size=5

    if tgt_marker is not None: # 255
        plotter.add_points(tgt_marker, color=[175, 175, 175], point_size=marker_size, render_points_as_spheres=True, opacity=marker_opacity)

    plotter.show()


def pc_normalize(pc, centroid=None, m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if m is None:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def ply2np_vox(xyz, voxel_size=2, scale=1.0):
    if type(xyz) is str:
        pcd = o3d.io.read_point_cloud(xyz)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_pts = np.asarray(downpcd.points)
    pcd_pts = pcd_pts / scale
    return pcd_pts

def norm_vox(xyz, voxel_size):
    pc, centroid, m = pc_normalize(xyz)
    pc_vox = ply2np_vox(pc, voxel_size=voxel_size, scale=1.0) * m + centroid
    return pc_vox, m

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

if __name__ == '__main__':

    root_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/in_silico/" # change to yours
    test_list = root_path + "Deform_mesh_npz_test/list.npz"
    test_path = root_path + "Deform_mesh_npz_test/Test/"

    test_list = np.load(test_list)['test']
    index = 612
    sigma = None
    rot = False

    """
    Load one npz and check
    
    src_vs: source point cloud/vertices, assumed to be obtained from preoperative
    src_f: source mesh faces
    tgt_vs: target point cloud, assumed to be obtained during intraoperative
    tgt_f: target mesh faces
    tgt_full: deformed src vs, sharing the same faces with the source
    src_markers: volumetric vertices of the source 
    tgt_markers: volumetric vertices of the target/deformed source
    """

    src_vs, src_f, tgt_vs, tgt_f, tgt_full, flow, src_markers, tgt_markers = load_test_from_npz(test_path+test_list[index], sigma=sigma, rot=rot)

    compare_mesh(src_vs=src_vs, src_faces=src_f)
    compare_mesh(src_vs=src_vs, src_faces=src_f, tgt_vs=tgt_full, tgt_faces=src_f)
    # compare_mesh(src_vs=src_vs_w, src_faces=src_f, tgt_vs=tgt_full, tgt_faces=src_f)

    src_vs, _ = norm_vox(src_vs, 0.04)
    tgt_vs, _ = norm_vox(tgt_vs, 0.04)

    R, t = weighted_procrustes(torch.from_numpy(src_markers).double().cuda(),
                               torch.from_numpy(tgt_markers).double().cuda())
    R = R.cpu().numpy()
    t = t.cpu().numpy()[:, np.newaxis]

    src_marker_pred = (np.matmul(R, src_markers.T) + t).T
    src_vs_pred = (np.matmul(R, src_vs.T) + t).T

    vis_pc_marker(src_vs=src_vs_pred, tgt_vs=tgt_vs)

    #vis_pc_marker(src_vs=src_marker_pred, tgt_vs=tgt_markers)
    #vis_pc_marker(tgt_vs=tgt_vs, tgt_marker=tgt_markers)
    #vis_pc_marker(tgt_marker=tgt_markers)
    #vis_pc_marker(tgt_marker=tgt_markers)
    #vis_pc_marker(src_vs=src_vs, src_marker=src_markers)
    #vis_pc_marker(src_marker=src_markers)
    #vis_pc_marker(src_marker=src_markers)
    #vis_mesh_pc(tgt_vs_noise, tgt_f, tgt_vs_noise)
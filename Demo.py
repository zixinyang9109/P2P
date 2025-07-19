import torch
from pointnet2_ops import pointnet2_utils
from p2p_util import matching, generate_node_clusters, index_select, weighted_procrustes, pairwise_distance
import numpy as np
import pyvista as pv


def vis_pc_marker_multi(data_list, titles=None, point_opacity=1, marker_opacity=0.6,
                        point_size=12, marker_size=10):
    """
    Visualize multiple point cloud registration results in a single window

    Args:
        data_list: List of dicts with keys 'src_vs', 'tgt_vs', 'src_marker', 'tgt_marker'
        titles: List of titles for each subplot
        point_opacity: Opacity for point clouds
        marker_opacity: Opacity for markers
        point_size: Size of point cloud points
        marker_size: Size of marker points
    """
    n_views = len(data_list)
    if titles is None:
        titles = [f"View {i + 1}" for i in range(n_views)]

    # Calculate figure width based on number of views and height
   # figure_width = figure_height * n_views  # Maintain aspect ratio

    # Create plotter with subplots and custom window size
    plotter = pv.Plotter(shape=(1, n_views))# , window_size=(figure_width, figure_height)
    plotter.set_background('white')

    # Define colors
    src_color = [0, 150, 255]  # Blue for source
    tgt_color = [254, 92, 92]  # Red for target
    marker_color = [175, 175, 175]  # Gray for markers

    # Common parameters
    point_params = {
        'point_size': point_size,
        'render_points_as_spheres': True,
        'opacity': point_opacity
    }

    marker_params = {
        'point_size': marker_size,
        'render_points_as_spheres': True,
        'opacity': marker_opacity
    }

    for i, data in enumerate(data_list):
        plotter.subplot(0, i)

        # Add source points
        if data.get('src_vs') is not None:
            plotter.add_points(data['src_vs'], color=src_color, **point_params)

        # Add target points
        if data.get('tgt_vs') is not None:
            plotter.add_points(data['tgt_vs'], color=tgt_color, **point_params)

        # Add source markers
        if data.get('src_marker') is not None:
            plotter.add_points(data['src_marker'], color=marker_color, **marker_params)

        # Add target markers
        if data.get('tgt_marker') is not None:
            plotter.add_points(data['tgt_marker'], color=marker_color, **marker_params)

        # Add title
        plotter.add_title(titles[i], font='courier', color='k', font_size=10)

    plotter.show()


def cal_error(gt, pred, print_error=True, label=""):
    """
    Calculate registration error metrics

    Args:
        gt: Ground truth points
        pred: Predicted points
        print_error: Whether to print the error
        label: Label for the error output

    Returns:
        diff_mean, diff_max, diff_std, RE (RMS-TRE)
    """
    # Convert to numpy if needed
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()

    # Calculate difference vector and norms
    diff_vec = pred - gt
    diff_norm = np.linalg.norm(diff_vec, axis=1)

    # Calculate statistics
    diff_mean = np.mean(diff_norm)
    diff_std = np.std(diff_norm)
    diff_max = np.max(diff_norm)

    # RMS TRE (Root Mean Square Target Registration Error)
    RE = np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1)))

    if print_error:
        prefix = f"{label}: " if label else ""
        print(f"{prefix}mean error: {diff_mean:.2f}, max: {diff_max:.2f}, "
              f"std: {diff_std:.2f}, RE: {RE:.2f}")

    return diff_mean, diff_max, diff_std, RE


def transform_points(points, rotation, translation):
    """
    Apply rigid transformation to points

    Args:
        points: Input points (numpy array)
        rotation: Rotation matrix
        translation: Translation vector

    Returns:
        Transformed points
    """
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(rotation):
        rotation = rotation.detach().cpu().numpy()
    if torch.is_tensor(translation):
        translation = translation.detach().cpu().numpy()

    # Ensure translation is properly shaped
    if translation.ndim == 1:
        translation = translation[:, None]

    return (np.matmul(rotation, points.T) + translation).T


def p2p(src_feats, tgt_feats, src_pcd, tgt_pcd, result=None, K=5):
    """
    Args:
        src_feats: Source point features
        tgt_feats: Target point features
        src_pcd: Source point cloud
        tgt_pcd: Target point cloud
        result: Result dictionary (will be created if None)
        K: Number of patches for patch-based registration

    Returns:
        Dictionary containing registration results
    """
    if result is None:
        result = {}

    # Initialize lists for storing results
    cl_R = []
    cl_t = []
    CP_res = []

    # Complete to partial registration
    conf_matrix_pred, match_pred = matching(src_feats.unsqueeze(0), tgt_feats.unsqueeze(0))

    corr_g = match_pred[:, 1:]
    src_id = corr_g[:, 0]
    tgt_id = corr_g[:, 1]

    rot_g, t_g = weighted_procrustes(src_pcd[src_id, :], tgt_pcd[tgt_id, :],
                                     conf_matrix_pred[0, src_id, tgt_id])

    # Calculate registration quality
    src_warp = torch.matmul(src_pcd, rot_g.T) + t_g.T
    dist_mat = pairwise_distance(tgt_pcd, src_warp)
    cp_res_g = torch.mean(dist_mat.min(dim=1).values)

    cl_R.append(rot_g)
    cl_t.append(t_g)
    CP_res.append(cp_res_g)

    # Patches to partial registration
    # 1) Visible Source Point Estimation
    sim_matrix = torch.einsum("sc,tc->st", src_feats, tgt_feats)  # Optimized einsum
    src_conf = sim_matrix.sum(dim=1)

    _, src_idx = src_conf.sort(descending=True, dim=0)
    tgt_len = len(tgt_pcd) if tgt_pcd.dim() > 1 else tgt_pcd.shape[0]
    vis_src_pcd = src_pcd[src_idx[:tgt_len], :]

    # 2) Patch Candidate Generation
    fps_idx = pointnet2_utils.furthest_point_sample(vis_src_pcd.unsqueeze(0), K)
    fps_idx = fps_idx.squeeze(0).long()
    src_node = vis_src_pcd[fps_idx]

    # 3) Patch Sampling and Feature Computation
    win_size = min(len(tgt_pcd), len(src_pcd))
    src_indices_cl = generate_node_clusters(src_pcd, src_node, point_limit=win_size)
    src_xyz_cl = index_select(src_pcd, src_indices_cl, dim=0)
    src_feats_cl = index_select(src_feats, src_indices_cl, dim=0)

    # Use expand instead of repeat for memory efficiency
    tgt_feats_cl = tgt_feats.unsqueeze(0).expand(K, -1, -1)

    # 4) Patch-to-Target Matching
    conf_matrix_pred_cl, match_pred_cl = matching(src_feats_cl, tgt_feats_cl)

    # Cluster-wise rigid registration
    cl_unique, cl_counts = torch.unique(match_pred_cl[:, 0], return_counts=True, dim=0)

    # Process each cluster
    for cl_i in cl_unique:
        # Use boolean indexing for efficiency
        mask = match_pred_cl[:, 0] == cl_i
        match_pred_i = match_pred_cl[mask]

        s_id, t_id = match_pred_i[:, 1], match_pred_i[:, 2]
        s_id_g = src_indices_cl[cl_i, s_id]

        src_xyz_cl_i = src_xyz_cl[cl_i, s_id]
        tgt_xyz_cl_i = tgt_pcd[t_id]
        conf_v = conf_matrix_pred_cl[cl_i, s_id, t_id]

        rot_cl_i, t_cl_i = weighted_procrustes(src_xyz_cl_i, tgt_xyz_cl_i, conf_v)

        # 5) Optimal Candidate Selection
        src_warp = torch.matmul(src_pcd, rot_cl_i.T) + t_cl_i.T
        dist_mat = pairwise_distance(tgt_pcd, src_warp)
        cp_res = torch.mean(dist_mat.min(dim=1).values)

        cl_R.append(rot_cl_i)
        cl_t.append(t_cl_i)
        CP_res.append(cp_res)

    # Find best registration
    if CP_res:
        min_index_cp = torch.stack(CP_res).argmin().item()
    else:
        min_index_cp = 0

    result.update({
        'R_cl': cl_R,
        't_cl': cl_t,
        'CP_res': CP_res,
        'best_index_cp': min_index_cp,
    })

    return result


if __name__ == '__main__':

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    # data = torch.load("./Data/3596_data.pt")
    # feats = torch.load("./Data/3596_feats.pt")

    # data = torch.load("./Data/725_data.pt")
    # feats = torch.load("./Data/725_feats.pt")

    data = torch.load("./Data/3506_data.pt")
    feats = torch.load("./Data/3506_feats.pt")

    # Extract data
    (src_pcd, tgt_pcd, src_marker, tgt_marker,
     s_c, t_c, scale, R_gt, t_gt) = (
        data['src_pcd'], data['tgt_pcd'], data['src_marker'], data['tgt_marker'],
        data['s_c'], data['t_c'], data['scale'], data['R_gt'], data['t_gt']
    )

    src_pcd_cuda = torch.from_numpy(src_pcd).to(device).float()
    tgt_pcd_cuda = torch.from_numpy(tgt_pcd).to(device).float()
    src_feats_cuda = feats['src_feats_cuda'].to(device)
    tgt_feats_cuda = feats['tgt_feats_cuda'].to(device)

    # Run registration
    print("Running point-to-point registration...")
    result = p2p(src_feats_cuda, tgt_feats_cuda, src_pcd_cuda, tgt_pcd_cuda, result={}, K=5)

    # Get registration results
    rot_cl = result['R_cl']
    trans_cl = result['t_cl']
    best_cl_index = result['best_index_cp']

    print(f"Best cluster index: {best_cl_index}")
    print("\n" + "=" * 60)

    # Calculate initial error (before registration)
    print("INITIAL POSITION (before registration):")
    _, _, _, initial_rms_tre = cal_error(tgt_marker * scale, src_marker * scale,
                                         print_error=True, label="Initial")

    # Prepare visualization data and RMS-TRE values
    vis_data = []
    rms_tre_values = []

    # 1. Initial position
    vis_data.append({
        'src_vs': src_pcd,
        'tgt_vs': tgt_pcd
    })
    rms_tre_values.append(initial_rms_tre)

    # 2. Complete-to-partial registration result
    if len(rot_cl) > 0:
        print("\nCOMPLETE-TO-PARTIAL REGISTRATION:")
        src_marker_pred_global = transform_points(src_marker, rot_cl[0], trans_cl[0])
        src_pcd_pred_global = transform_points(src_pcd, rot_cl[0], trans_cl[0])

        _, _, _, global_rms_tre = cal_error(tgt_marker * scale, src_marker_pred_global * scale,
                                            print_error=True, label="Complete-to-partial")

        vis_data.append({
            'src_vs': src_pcd_pred_global,
            'tgt_vs': tgt_pcd
        })
        rms_tre_values.append(global_rms_tre)

    # 3. Patches-to-partial registration result
    if len(rot_cl) > best_cl_index:
        print("\nPATCHES-TO-PARTIAL REGISTRATION:")
        src_marker_pred_patch = transform_points(src_marker, rot_cl[best_cl_index], trans_cl[best_cl_index])
        src_pcd_pred_patch = transform_points(src_pcd, rot_cl[best_cl_index], trans_cl[best_cl_index])

        _, _, _, patch_rms_tre = cal_error(tgt_marker * scale, src_marker_pred_patch * scale,
                                           print_error=True, label="Patches-to-partial")

        vis_data.append({
            'src_vs': src_pcd_pred_patch,
            'tgt_vs': tgt_pcd
        })
        rms_tre_values.append(patch_rms_tre)

    print("=" * 60)

    # Create titles with RMS-TRE values
    base_titles = ["Initial Position", "Complete-to-Partial", "Patches-to-Partial"]
    titles_with_rms = []

    for i, (title, rms_tre) in enumerate(zip(base_titles[:len(vis_data)], rms_tre_values)):
        titles_with_rms.append(f"{title}\nRMS-TRE: {rms_tre:.2f}")

    print(f"\nShowing {len(vis_data)} registration results with RMS-TRE values...")
    # You can adjust the figure height here (default is 600 pixels)
    vis_pc_marker_multi(vis_data, titles=titles_with_rms)

    print("Visualization complete!")
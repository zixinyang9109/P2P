import numpy as np

import matplotlib.pyplot as plt

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

    file = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/stat_svd.npz"
    sigma = "None" # Noise level

    folders = ["P2P", "LiverMatch", "Lepard", "RegTR", "RoITR"]
    folder = folders[0]
    # add_name = "004002"
    # add_name = "_w_p2p_Lepard"
    add_name = "_w_p2p_LiverMatch"

    result_file = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Eva_in_silico/Results_in_silico/" + folder + "/" + sigma +add_name+ ".npy"

    data = np.load(file)
    vis = data['vis']#[:1000]
    deform = data['deform']#[:1000]

    result = np.load(result_file)#['re']
    #result = deform

    center_vis = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bins_vis = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("mean")
    print("&", round(np.mean(result), 2), "$\pm$", round(np.std(result), 2), "\n ")

    _, mean_reg_v, std_reg_v = stat(vis, result, bins_vis)
    print("vis")
    for i in np.arange(len(mean_reg_v)):
        print("&", round(mean_reg_v[i], 2), "$\pm$", round(std_reg_v[i], 2), "\n ")






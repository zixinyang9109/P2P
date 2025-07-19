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

    #file ="/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/stat.npz"
    file = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/stat_svd.npz"
    fig_path = "/home/yzx/Pictures/Visualization/"

    data = np.load(file)
    vis = data['vis']
    deform = data['deform']#[:, 0]
    max_deform = data['max_deform']
    init = data['init']

    bins_vis = [0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1]
    center_vis = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    bins_def = [0.5 + i * 4 for i in np.arange(5)]
    center_def = [i * 4 for i in np.arange(4)]

    # Alternative approach: Create three separate figures
    plt.figure(figsize=(3.75, 2.5), dpi=400)
    plt.hist(vis, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel('Visibility ratio', fontsize=10)
    plt.ylabel('Number', fontsize=10)
    plt.xlim([0.2, 1])
    plt.tight_layout(pad=1.5)
    plt.show()
    # plt.savefig(fig_path+'/fig1_vis.png')

    plt.figure(figsize=(3.75, 2.5), dpi=400)
    plt.hist(deform, range=(0, 12), bins=6, edgecolor='black', alpha=0.7)
    plt.xlabel('RMS-TRE (mm)', fontsize=10)
    plt.ylabel('Number', fontsize=10)
    plt.xlim([0, 12])
    plt.tight_layout(pad=1.5)
    plt.show()
    # plt.savefig(fig_path+'/fig2_deform.png')

    plt.figure(figsize=(3.75, 2.5), dpi=400)
    plt.hist(max_deform, range=(0, 40), bins=6, edgecolor='black', alpha=0.7)
    plt.xlabel('Max-TRE (mm)', fontsize=10)
    plt.ylabel('Number', fontsize=10)
    plt.xlim([0, 40])
    plt.tight_layout(pad=1.5)
    plt.show()
    # plt.savefig(fig_path+'/fig3_max.png')

    plt.figure(figsize=(3.75, 2.5), dpi=400)
    plt.hist(init, range=(20, 120), bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel('RMS-TRE (mm)', fontsize=10)
    plt.ylabel('Number', fontsize=10)
    plt.xlim([20, 120])
    plt.tight_layout(pad=1.5)
    plt.show()
    # plt.savefig(fig_path + '/fig4_init.png')
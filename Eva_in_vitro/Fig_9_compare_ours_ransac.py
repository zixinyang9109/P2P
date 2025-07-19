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

    file = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/in_vitro/stat.npz"

    result_file_0 = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Eva_in_vitro/Results_in_vitro/Result/Lepard/0.npy"

    result_file_1 = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Eva_in_vitro/Results_in_vitro/Result_ransac/Lepard/0.npy"

    result_file_2 = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Eva_in_vitro/Results_in_vitro/Result/P2P/5_Lepard_cl.npy"

    data = np.load(file)
    vis = data['vis']
    deform = data['deform']  # [:, 0]

    result_0 = np.load(result_file_0)
    result_1 = np.load(result_file_1)
    result_2 = np.load(result_file_2)


    center_vis = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bins_vis = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    _, mean_vis_0, std_vis_0 = stat(vis, result_0, bins_vis)
    _, mean_vis_1, std_vis_1 = stat(vis, result_1, bins_vis)
    _, mean_vis_2, std_vis_2 = stat(vis, result_2, bins_vis)

    SMALL_SIZE = 12 *1.5
    MEDIUM_SIZE = 12*2
    BIGGER_SIZE = 12*2

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(center_vis, mean_vis_0, 'o-', label="Lepard", markersize=SMALL_SIZE, color='orange')
    ax.plot(center_vis, mean_vis_1, 'o-', label= "Lepard + RANSAC ICP", markersize=SMALL_SIZE, color='red')
    ax.plot(center_vis, mean_vis_2, 'o-', label="Lepard + proposed P2P", markersize=SMALL_SIZE, color='blue')


    # Set labels and title
    ax.set_xlabel('Visibility Ratio')
    ax.set_ylabel('RMS-TRE (mm)')
    #ax.set_title('Scatter Plot')
    ax.set_ylim(0, 25)

    # Add legend
    ax.legend(loc='upper right')

    # Show plot
    plt.show()


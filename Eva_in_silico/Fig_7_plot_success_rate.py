import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def calculate_success_rate(rms_tre_values, threshold):

    # Count successful registrations (RMS-TRE < threshold)
    successful_count = np.sum((rms_tre_values<threshold)*1.0)

    # Calculate success rate as percentage
    total_count = len(rms_tre_values)
    success_rate = (successful_count / total_count) * 100.0

    return success_rate

def plot_threshold_vs_success_rate(rms_tre_data, method_names=None, thresholds=None):
    """
    Create a plot comparing multiple methods in terms of threshold vs. success rate.

    Parameters:
    rms_tre_data (dict or list): Dictionary mapping method names to lists of RMS-TRE values,
                                 or list of lists containing RMS-TRE values for each method
    method_names (list): List of method names (used if rms_tre_data is a list)
    thresholds (list, optional): List of threshold values to evaluate.
                                 If None, a range of values will be automatically generated.

    Returns:
    fig, ax: The matplotlib figure and axis objects
    """
    # Convert list input to dictionary if necessary
    if isinstance(rms_tre_data, list):
        if method_names is None:
            raise ValueError("method_names must be provided when rms_tre_data is a list")
        rms_tre_dict = {method: values for method, values in zip(method_names, rms_tre_data)}
    else:
        rms_tre_dict = rms_tre_data

    # Generate thresholds if not provided
    if thresholds is None:
        # Find min and max values across all methods
        all_values = [val for values in rms_tre_dict.values() for val in values]
        min_tre = min(all_values)
        max_tre = max(all_values)

        # Create threshold range from near 0 to slightly above maximum value
        thresholds = np.linspace(min_tre * 0.1, max_tre * 1.2, 100)

    SMALL_SIZE = 12 *1.5
    MEDIUM_SIZE = 12 * 2
    BIGGER_SIZE = 12 * 2

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color and line style options
    colors = ['red', 'blue']
    makers = ['o-', 'X-']

    # Calculate and plot success rate curves for each method
    # Special color assignment for related methods
    method_colors = {}

    # Check if the specific methods are in the data
    methods = list(rms_tre_dict.keys())

    # Assign colors - Methods A and B share a color, Methods C and D share a color
    for method in methods:
        if "LiverMatch" in method or "LiverMatch" in method:
            method_colors[method] = colors[0]  # First color for Method A and B
        elif "Lepard" in method or "Lepard" in method:
            method_colors[method] = colors[1]  # Second color for Method C and D


    # Plot each method
    for i, (method_name, rms_tre_values) in enumerate(rms_tre_dict.items()):
        success_rates = [calculate_success_rate(rms_tre_values, th) for th in thresholds]

        # Determine line style - solid line for base methods (A, C), dashed for methods with P2P (B, D)
        if "P2P" in method_name:
            line_style = makers[1]  # Dashed for methods with P2P
        else:
            line_style = makers[0]  # Solid for base methods

        ax.plot(thresholds, success_rates,
                line_style,
                alpha=0.6,
                markersize=SMALL_SIZE*0.6,
                label=method_name,
                color=method_colors[method_name],
                linewidth=1.5)

    # Customize the plot
    ax.set_xlabel(r'$\tau$ (mm)')
    ax.set_ylabel('Success Rate (%)')

    # Set y-axis to percentage
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))

    # Add grid for readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend(loc='lower right')

    # Set axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)  # Set to slightly above 100% for better visualization

    plt.tight_layout()
    return fig, ax


# Example usage:
if __name__ == "__main__":
    file = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/stat.npz"
    sigma = "None"
    folder = "P2P"
    result_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Eva_in_silico/Results_in_silico/"
    result_file_0 = result_path + folder + "/" + sigma + "_wo_p2p_LiverMatch.npy"
    result_file_1 = result_path + folder + "/" + sigma + "_w_p2p_LiverMatch.npy"
    result_file_2 = result_path + folder + "/" + sigma + "_wo_p2p_Lepard.npy"
    result_file_3 = result_path + folder + "/" + sigma + "_w_p2p_Lepard.npy"

    data = np.load(file)
    vis = data['vis']
    deform = data['deform']  # [:, 0]

    result_0 = np.load(result_file_0)[vis<0.3]
    result_1 = np.load(result_file_1)[vis<0.3]
    result_2 = np.load(result_file_2)[vis<0.3]
    result_3 = np.load(result_file_3)[vis<0.3]

    # Create a dictionary of methods
    methods_data = {
        "LiverMatch": result_0,
        "LiverMatch + proposed P2P": result_1,
        "Lepard": result_2,
        "Lepard + proposed P2P": result_3
    }

    # Define custom thresholds if desired
    custom_thresholds = np.linspace(0, 25, 50)

    # Create the plot
    fig, ax = plot_threshold_vs_success_rate(methods_data, thresholds=custom_thresholds)

    # Save the figure (optional)
    # plt.savefig('threshold_vs_success_rate.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np


def merge_images_vertically(image_paths, output_path, spacing=0.05):
    """
    Merge multiple images vertically into a single figure with proper spacing.

    Parameters:
    -----------
    image_paths : list of str
        List of paths to the images to be merged
    output_path : str
        Path to save the merged figure
    spacing : float
        Amount of vertical space between images (as a fraction of figure height)
    """
    # Read all images
    images = [mpimg.imread(img_path) for img_path in image_paths]

    # Create figure with proper height ratio
    fig = plt.figure(figsize=(8, 3 * len(images)), dpi=400)

    # Create grid for subplots with proper spacing
    gs = GridSpec( 1, len(images), figure=fig, hspace=spacing)

    # Add each image to the figure
    for i, img in enumerate(images):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.axis('off')  # Turn off axis

    # Save combined figure
    plt.savefig(output_path, bbox_inches='tight', dpi=400)
    plt.close()

    print(f"Merged image saved to {output_path}")

if __name__ == '__main__':

    # Example usage
    fig_path = "/home/yzx/Pictures/Visualization/"
    image_paths = [fig_path + 'fig1_vis.png', fig_path + 'fig2_deform.png', fig_path + 'fig3_max.png']
    output_path = fig_path +  'data_property.png'
    merge_images_vertically(image_paths, output_path, spacing=0.0)
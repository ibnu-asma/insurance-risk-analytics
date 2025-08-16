import os
import matplotlib.pyplot as plt

def save_plot(filename: str, output_dir: str = 'plots'):
    """Save a plot to the specified directory.
    
    Args:
        filename (str): Name of the plot file.
        output_dir (str): Directory to save the plot (default: 'plots').
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()
import os
import numpy as np

def load_data(filename: str) -> np.ndarray:
    """
    Load data from a .npy file.

    Parameters:
    filename (str): The name of the file to load.

    Returns:
    np.ndarray: The data loaded from the specified file.
    """
    return np.load(file=filename)

def get_absolute_path(file_name: str) -> str:
    """
    Generate the absolute path for a file located in the 'data' directory.
    Assumes that the 'data' directory is at the same level as the 'src' directory.

    Parameters:
    file_name (str): The name of the file for which to get the absolute path.

    Returns:
    str: The absolute path to the specified file.

    Raises:
    FileNotFoundError: If the specified file does not exist in the 'data' directory.
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Construct the path to the 'data' directory
    data_dir = os.path.join(project_root, "data")
    
    # Construct the full file path
    file_path = os.path.join(data_dir, file_name)
    
    # Check if the file exists
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"{file_name} not found in {data_dir}")
    

def get_plot_save_path(image_name: str) -> str:
    """
    Generate the absolute path for saving a plot in the 'plots' directory.
    Assumes that the 'plots' directory is at the same level as the 'src' directory.

    Parameters:
    image_name (str): The name of the image file to save.

    Returns:
    str: The absolute path to the specified image file in the 'plots' directory.
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Construct the path to the 'plots' directory
    plots_dir = os.path.join(project_root, "plots")
    
    # Construct the full file path
    file_path = os.path.join(plots_dir, image_name)
    
    # Create the plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    return file_path

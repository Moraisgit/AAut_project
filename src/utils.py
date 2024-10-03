import os
import numpy as np
from colorama import init, Fore


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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Construct the path to the 'data' directory
    data_dir = os.path.join(project_root, "data")

    # Construct the full file path
    file_path = os.path.join(data_dir, file_name)

    # Check if the file exists
    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"{file_name} not found in {data_dir}")


def save_npy_to_output(file_name: str, data: np.array) -> None:
    """
    Save a NumPy array to a .npy file in the 'output' directory of the project.

    Parameters:
    file_name (str): The name to save the file as, including the .npy extension.
    data (np.array): The NumPy array to be saved.
    
    Returns:
    None
    """
    # Initialize colorama to use colored output in the terminal
    init()

    # Get the project root directory by navigating one level up from the current script's directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Construct the path to the 'output' directory within the project root
    output_dir = os.path.join(project_root, "output")

    # If the 'output' directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the full file path by combining the output directory and the provided file name
    file_path = os.path.join(output_dir, file_name)

    # Save the provided NumPy array to a .npy file at the constructed file path
    np.save(file_path, data)

    # Print the full path to the saved file using blue text for the file path
    print(f"File saved at: {Fore.BLUE}{file_path}{Fore.RESET}")


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
    project_root = os.path.abspath(path=os.path.join(os.path.dirname(__file__), ".."))

    # Construct the path to the 'plots' directory
    plots_dir = os.path.join(project_root, "plots")

    # Construct the full file path
    file_path = os.path.join(plots_dir, image_name)

    # Create the plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    return file_path
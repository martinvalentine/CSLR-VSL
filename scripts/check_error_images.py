import os
import concurrent.futures
from PIL import Image
from tqdm import tqdm
import platform # Import platform module

# --- Function to check a single image ---
# (This function remains the same as the previous parallel version)
def check_image(img_path):
    """
    Tries to open and load an image file.
    Returns None if successful, or a tuple (img_path, error_message) if corrupted.
    """
    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            img.load()
        return None
    except Exception as e:
        return (img_path, str(e))

# --- Main function modified for parallelism with core reservation ---
def find_corrupted_images_parallel(root_dir, extensions=(".jpg", ".jpeg", ".png"), max_workers=None):
    """
    Finds corrupted images in a directory using multiple CPU cores,
    attempting to leave one core free if max_workers is not specified.

    Args:
        root_dir (str): The root directory to search.
        extensions (tuple): Tuple of file extensions to check (lowercase).
        max_workers (int, optional): Explicitly set the maximum number of worker processes.
                                     If None, calculates based on CPU cores minus one.

    Returns:
        list: A list of tuples, where each tuple contains (image_path, error_message)
              for corrupted images.
    """
    corrupted = []
    all_images = []

    print(f"Scanning for images in: {root_dir}")
    # Walk all subdirectories to find image files
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                all_images.append(os.path.join(root, file))

    if not all_images:
        print("No images found with the specified extensions.")
        return []

    print(f"Found {len(all_images)} images. Checking integrity using parallel processing...")

    # Determine the number of workers to use
    workers_to_use = max_workers
    if workers_to_use is None:
        total_cores = os.cpu_count()
        if total_cores:
            # Leave one core free, but always use at least 1 worker
            workers_to_use = max(1, total_cores - 1)
            print(f"--> Detected {total_cores} cores. Using {workers_to_use} worker processes (leaving 1 core free for system tasks).")
        else:
            # Fallback if cpu_count() fails
            workers_to_use = 1
            print("--> Could not automatically determine CPU count, using 1 worker process.")
    else:
         print(f"--> Using explicitly specified {workers_to_use} worker processes.")


    print(f"Starting image integrity check with {workers_to_use} workers...\n")

    # Use ProcessPoolExecutor with the calculated number of workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_to_use) as executor:
        results = list(tqdm(executor.map(check_image, all_images), total=len(all_images), desc="Checking Images"))

    # Filter out the None results (good images) and collect corrupted ones
    corrupted = [result for result in results if result is not None]

    return corrupted

# --- Main execution block ---
if __name__ == "__main__":
    # Print system info for context
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")

    # IMPORTANT: Put your code that uses multiprocessing inside this block!
    folder = "/home/martinvalentine/Desktop/CSLR-VSL/data/interim/frames/VSL_V1_5"  # TODO: <--- CHANGE THIS TO YOUR FOLDER PATH
    # folder = "path/to/your/image_folder" # Example placeholder

    if not os.path.isdir(folder):
         print(f"\nError: Folder not found at {folder}")
         print("Please update the 'folder' variable in the script.")
    else:
        print(f"\nChecking for corrupted JPEGs in: {folder}")

        # Set max_workers=None to use the "cores - 1" logic.
        # Or set max_workers to a specific number (e.g., 4) to override.
        explicit_workers = None # Set to None for automatic calculation, or e.g., 4 to force 4 workers

        bad_images = find_corrupted_images_parallel(folder, max_workers=explicit_workers)

        if bad_images:
            print(f"\nFound {len(bad_images)} corrupted images:")
            # Limit printing if there are too many errors
            max_errors_to_print = 50
            for i, (path, error) in enumerate(bad_images):
                 if i < max_errors_to_print:
                     print(f"  {path} -- {error}")
                 elif i == max_errors_to_print:
                     print(f"  ... and {len(bad_images) - max_errors_to_print} more.")
                     break
        else:
            print("\nNo corrupted images found.")
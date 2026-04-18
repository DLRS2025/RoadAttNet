import os
import glob
import logging
import argparse
from tqdm import tqdm
import concurrent.futures
import multiprocessing
from config import load_config
from Feature_Extration import feature_extraction

def setup_logger():
    # Set up a logger that writes to both file and console
    log_file = "batch_extraction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--workers", type=int, default=0, help="Number of CPU cores to use. Default is 0 (auto-detect all cores).")
    return p.parse_args()


def process_single_image(args):
    """Helper function to process a single image in a separate process."""
    img_path, feature2_dir, feature1_dir = args
    try:
        # feature2_dir -> cwl, feature1_dir -> blurred
        feature_extraction(img_path, feature2_dir, feature1_dir)
        return True, img_path, None
    except Exception as e:
        return False, img_path, str(e)


def main():
    args = parse_args()
    logger = setup_logger()
    logger.info("Initializing Feature Extraction pipeline...")
    
    # Load configuration
    cfg = load_config(args.config)
    
    rgb_dir = cfg.rgb_dir
    feature1_dir = cfg.feature1_dir # blurred
    feature2_dir = cfg.feature2_dir # cwl
    
    logger.info("Loaded config paths:")
    logger.info(f"  RGB Source: {rgb_dir}")
    logger.info(f"  Blurred Output: {feature1_dir}")
    logger.info(f"  CWL Output: {feature2_dir}")

    # Create directories if they do not exist
    os.makedirs(feature1_dir, exist_ok=True)
    os.makedirs(feature2_dir, exist_ok=True)
    
    # Get all TIFF images from the source directory
    tiff_files = glob.glob(os.path.join(rgb_dir, "*.tiff")) + glob.glob(os.path.join(rgb_dir, "*.tif"))
    
    if not tiff_files:
        logger.error(f"No TIFF images found in {rgb_dir}. Please check your configuration.")
        return

    # Determine number of workers
    num_cores = multiprocessing.cpu_count()
    workers = args.workers if args.workers > 0 else max(1, num_cores - 1) # leave 1 core free to prevent UI freeze
    
    logger.info(f"Found {len(tiff_files)} images. Starting feature extraction using {workers} CPU cores...")
    
    # Prepare arguments for the worker pool
    process_args = [(img_path, feature2_dir, feature1_dir) for img_path in tiff_files]
    
    success_count = 0
    error_count = 0
    
    # Process images concurrently
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Wrap the executor.map with tqdm for the progress bar
        results = list(tqdm(executor.map(process_single_image, process_args), total=len(process_args), desc="Extracting Features"))
        
        for success, img_path, error_msg in results:
            if success:
                success_count += 1
            else:
                logger.error(f"Error processing {img_path}: {error_msg}")
                error_count += 1
            
    logger.info(f"Extraction completed! Successfully processed: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()

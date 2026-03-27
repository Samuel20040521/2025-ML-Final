import pandas as pd
import os
from PIL import Image
import io
import argparse
import glob

def process_and_delete_parquet(parquet_path, output_dir):
    """
    Process a single parquet file and delete it after successful processing.
    
    Parameters:
    parquet_path (str): Path to the Parquet file
    output_dir (str): Directory where images will be saved
    
    Returns:
    bool: True if processing was successful, False otherwise
    """
    try:
        # Read the Parquet file
        df = pd.read_parquet(parquet_path)
        
        for _, row in df.iterrows():
            # Get the image data and filename
            image_data = row['image']['bytes']
            filename = row['image']['path']
            
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(image_data))
            
            # Create filepath and save the image
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
        
        # Delete the parquet file after successful processing
        os.remove(parquet_path)
        print(f"Processed and deleted: {parquet_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {parquet_path}: {str(e)}")
        return False

def process_parquet_folder(input_folder, output_dir):
    """
    Process all parquet files in a folder.
    
    Parameters:
    input_folder (str): Path to the folder containing parquet files
    output_dir (str): Directory where images will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all parquet files in the input folder
    parquet_files = glob.glob(os.path.join(input_folder, "*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_folder}")
        return
    
    total_files = len(parquet_files)
    successful = 0
    failed = 0
    
    for parquet_file in parquet_files:
        if process_and_delete_parquet(parquet_file, output_dir):
            successful += 1
        else:
            failed += 1
        
        # Print progress
        print(f"Progress: {successful + failed}/{total_files} files processed")
    
    # Print final summary
    print("\nProcessing complete!")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process parquet files from folder and save images')
    
    # Add arguments
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Folder containing parquet files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the images')

    # Parse arguments
    args = parser.parse_args()

    try:
        process_parquet_folder(args.input_folder, args.output_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
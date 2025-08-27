import os
import shutil
import pandas as pd
import logging
import streamlit as st
import pangaeapy.pandataset as pdataset
import xarray as xr
import requests
import zipfile
import re
import uuid
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch dataset details using pangaeapy
def fetch_dataset_details(doi):
    try:
        dataset = pdataset.PanDataSet(id=doi)
        dataset.setMetadata()
        abstract = getattr(dataset, 'abstract', "No description available") or "No description available"
        param_dict = dataset.getParamDict()
        short_names = param_dict.get('shortName', [])
        parameters = ', '.join(short_names) + "..." if len(short_names) > 10 else ', '.join(short_names)

        return abstract, parameters

    except Exception as e:
        logging.error(f"Error fetching dataset details for DOI {doi}: {e}")
        return "No description available", "No parameters available"

# Conversion function
def convert_df_to_csv(df):
    """
    Convert a dataset to CSV format. Handles different input types:
    - pandas DataFrame: calls to_csv() directly
    - string (sandbox path): tries to load a CSV file from the path
    - other types: returns empty bytes
    
    Returns:
        bytes: CSV data encoded in UTF-8 or empty bytes if conversion fails
    """
    logging.debug(f"Converting data to CSV, type: {type(df)}")
    
    if isinstance(df, pd.DataFrame):
        # If it's already a DataFrame, just convert it
        return df.to_csv().encode('utf-8')
    elif isinstance(df, str) and os.path.isdir(df):
        # If it's a directory path (sandbox), try to find and read a CSV file
        try:
            csv_files = [f for f in os.listdir(df) if f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(df, csv_files[0])
                csv_df = pd.read_csv(csv_path)
                return csv_df.to_csv().encode('utf-8')
            else:
                logging.warning(f"No CSV files found in directory: {df}")
                return b''
        except Exception as e:
            logging.error(f"Error reading CSV from sandbox: {e}")
            return b''
    elif isinstance(df, xr.Dataset):
        # For xarray datasets, convert to DataFrame first
        try:
            df_converted = df.to_dataframe()
            return df_converted.to_csv().encode('utf-8')
        except Exception as e:
            logging.error(f"Error converting xarray Dataset to CSV: {e}")
            return b''
    else:
        # For unsupported types
        logging.warning(f"Unsupported data type for CSV conversion: {type(df)}")
        return b''

# Direct download function using static download link
def download_pangaea_dataset(doi_url, output_dir):
    """
    Downloads a PANGAEA dataset. It first tries to find a static download link.
    If that fails, it falls back to appending '?format=zip' to the DOI URL.
    
    Args:
        doi_url (str): The DOI URL (e.g., "https://doi.org/10.1594/PANGAEA.785092")
        output_dir (str): Directory where files will be saved
        
    Returns:
        tuple: (success boolean, path to the extracted files or error message)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize the URL to the doi.pangaea.de format
    if "doi.org" in doi_url:
        pangaea_url = doi_url.replace("doi.org", "doi.pangaea.de")
    else:
        pangaea_url = doi_url
    
    # --- METHOD 1: Find the 'static-download-link' on the page ---
    logging.info(f"Step 1: Attempting to find static download link at {pangaea_url}")
    try:
        response = requests.get(pangaea_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        static_link_element = soup.find('a', id='static-download-link')
        
        if static_link_element and static_link_element.get('href'):
            download_url = static_link_element.get('href')
            logging.info(f"Step 1 SUCCESS: Found static download URL: {download_url}")
        else:
            download_url = None
            logging.warning("Step 1 FAILED: Could not find the 'static-download-link' element.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Step 1 ERROR: Failed to fetch the landing page: {str(e)}")
        download_url = None

    # --- METHOD 2: Fallback to the '?format=zip' method ---
    if not download_url:
        logging.info("Step 2: Falling back to direct download method using '?format=zip'")
        # Ensure the URL doesn't already have query parameters
        base_pangaea_url = pangaea_url.split('?')[0]
        download_url = f"{base_pangaea_url}?format=zip"
        logging.info(f"Step 2: Constructed fallback URL: {download_url}")

    # --- Step 3: Proceed with downloading from the determined URL ---
    try:
        filename = "download.zip" # Default filename for the direct method
        # Try to get a better filename from the URL if possible
        if '?' in download_url:
             # For URLs like .../PANGAEA.908815?format=zip, create a filename
            doi_part = pangaea_url.split('/')[-1].split('?')[0]
            filename = f"{doi_part}.zip"
        else:
            filename = os.path.basename(download_url)
        
        local_file_path = os.path.join(output_dir, filename)
        
        logging.info(f"Step 3: Downloading file from {download_url} to {local_file_path}")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_file_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # --- Step 4: Handle Extraction ---
        if local_file_path.lower().endswith('.zip'):
            logging.info(f"Step 4: Extracting ZIP file {local_file_path} into {output_dir}")
            
            with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                # Extract all files directly into the output directory
                zip_ref.extractall(output_dir)
            
            logging.info(f"Successfully extracted files to {output_dir}")
            os.remove(local_file_path)
            logging.info(f"Removed temporary ZIP file {local_file_path}")
            return True, output_dir
        else:
            # If it's not a zip, the single file is the result
            logging.info(f"Successfully downloaded single file to {local_file_path}")
            return True, local_file_path
            
    except requests.exceptions.RequestException as e:
        error_message = f"ERROR: Download failed from URL {download_url}. Reason: {str(e)}"
        logging.error(error_message)
        return False, error_message
    except zipfile.BadZipFile:
        error_message = f"ERROR: The downloaded file from {download_url} is not a valid ZIP file."
        logging.error(error_message)
        return False, error_message
    except Exception as e:
        error_message = f"ERROR: An unexpected error occurred during download/extraction. Reason: {str(e)}"
        logging.error(error_message, exc_info=True)
        return False, error_message

# Updated fetch_dataset function
def fetch_dataset(doi, target_dir=None):
    """
    Enhanced function to fetch dataset files from a PANGAEA DOI and populate a specified directory.
    
    Args:
        doi (str): The DOI of the dataset (e.g., "https://doi.org/10.1594/PANGAEA.785092").
        target_dir (str, optional): Directory to save the dataset files. If None, creates a new sandbox.
    
    Returns:
        tuple: (dataset_path, dataset_name) where dataset_path is the path to the dataset files
               (or None if failed), and dataset_name is the dataset title.
    """
    logging.info(f"Starting fetch_dataset for DOI: {doi}")

    # If target_dir not provided, create a new sandbox (for backward compatibility)
    if target_dir is None:
        unique_id = str(uuid.uuid4())
        target_dir = os.path.join("tmp", "sandbox", unique_id)
        os.makedirs(target_dir, exist_ok=True)
        logging.info(f"Created sandbox directory: {target_dir}")
    else:
        os.makedirs(target_dir, exist_ok=True)
        logging.info(f"Using provided target_dir: {target_dir}")

    # Check cache
    if doi in st.session_state.datasets_cache:
        dataset, name = st.session_state.datasets_cache[doi]
        logging.info(f"Returning cached dataset for DOI {doi}, path: {dataset}")
        return dataset, name

    # Parse DOI to get dataset ID
    doi_match = re.search(r'PANGAEA\.(\d+)', doi)
    dataset_id = doi_match.group(1) if doi_match else doi.split('/')[-1].strip(')')
    dataset_name = f"PANGAEA.{dataset_id}"

    try:
        # Try loading as DataFrame via pangaeapy
        try:
            ds = pdataset.PanDataSet(int(dataset_id))
            dataset_name = ds.title
            base_data = ds.data
            if base_data is not None and not base_data.empty:
                csv_path = os.path.join(target_dir, "data.csv")
                base_data.to_csv(csv_path, index=False)
                logging.info(f"Saved DataFrame to {csv_path}")
                st.session_state.datasets_cache[doi] = (target_dir, dataset_name)
                return target_dir, dataset_name
            else:
                logging.info(f"No DataFrame via pangaeapy for DOI {doi}, proceeding to download files")
        except Exception as e:
            logging.warning(f"Pangaeapy approach failed for DOI {doi}: {str(e)}")

        # Normalize DOI URL for download
        if doi.startswith("10.1594/"):
            doi_url = f"https://doi.pangaea.de/{doi}"
        elif doi.startswith("PANGAEA."):
            doi_url = f"https://doi.pangaea.de/10.1594/{doi}"
        elif doi.startswith("https://doi.org/"):
            doi_url = doi
        elif not doi.startswith(("http://", "https://")):
            doi_url = f"https://doi.pangaea.de/10.1594/PANGAEA.{dataset_id}"
        else:
            doi_url = doi

        # Download files into target_dir
        success, result_path = download_pangaea_dataset(doi_url, target_dir)
        if success:
            logging.info(f"Successfully downloaded dataset to: {result_path}")
            st.session_state.datasets_cache[doi] = (target_dir, dataset_name)
            return target_dir, dataset_name
        else:
            logging.warning(f"Download failed: {result_path}")
            st.session_state.datasets_cache[doi] = (None, f"Failed: {result_path}")
            return None, f"Failed: {result_path}"

    except Exception as e:
        logging.error(f"Error in fetch_dataset for DOI {doi}: {str(e)}")
        st.session_state.datasets_cache[doi] = (None, f"Failed: {str(e)}")
        return None, f"Failed: {str(e)}"
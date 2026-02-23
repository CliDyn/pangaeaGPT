# search_pg_default.py
import requests
import pandas as pd
import logging
import re
from typing import List, Optional
from bs4 import BeautifulSoup
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pangaeapy.pandataset as pdataset
from .dataset_utils import fetch_dataset_details
from .fast_pangaea_client import fetch_metadata_fast, format_parameters, format_feature_tags
from ..utils.workspace import WorkspaceManager

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check if a variable is of a specific type
def check_if(x, cls):
    if x is not None and not isinstance(x, cls):
        raise TypeError(f"{x} must be of class: {', '.join([str(c) for c in cls])}")

# Utility functions equivalent to R functions
def pgc(x):
    return {k: v for k, v in x.items() if v is not None}

def strextract(string, pattern):
    match = re.search(pattern, string)
    return match.group(0) if match else None


# Function to parse the result
def parse_res(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    citation_tag = soup.select_one('div.citation a')
    citation = citation_tag.get_text(strip=True) if citation_tag else None

    supp_tag = soup.select_one('tr:contains("Supplement to:") .content')
    supp = supp_tag.get_text(strip=True) if supp_tag else None

    size_tag = soup.select_one('tr:contains("Size:") .content')
    size = size_tag.get_text(strip=True) if size_tag else None

    size_val = strextract(size, r"[0-9]+") if size else None
    meas = strextract(size, r"[A-Za-z].+") if size else None

    param_tags = soup.select('tr:contains("Parameter") .content')
    parameters = ', '.join([tag.text for tag in param_tags[:10]])
    if len(param_tags) > 10:
        parameters += "..."

    # Extract file URLs
    file_tags = soup.select('a[title="Download data"]')
    file_urls = [tag['href'] for tag in file_tags]

    return {
        'size': int(size_val.replace(",", "")) if size_val else None,
        'size_measure': meas,
        'citation': citation,
        'supplement_to': supp,
        'parameters': parameters,
        'file_urls': file_urls  # Always return a list
    }

# Main search function
def pg_search_default(query: str, count: int = 30, from_idx: int = 0, topic: Optional[str] = None,
                      mindate: Optional[str] = None, maxdate: Optional[str] = None,
                      minlat: Optional[float] = None, maxlat: Optional[float] = None,
                      minlon: Optional[float] = None, maxlon: Optional[float] = None, **kwargs) -> pd.DataFrame:
    # Type checking for all parameters
    check_if(count, (int,))
    check_if(topic, (str,))
    check_if(mindate, (str,))
    check_if(maxdate, (str,))
    check_if(minlat, (float, int))  # Allow integers as well
    check_if(maxlat, (float, int))
    check_if(minlon, (float, int))
    check_if(maxlon, (float, int))

    # Build parameters dictionary, converting floats to strings for API
    params = pgc({
        'q': query,
        'count': count,
        'offset': from_idx,
        'topic': topic,
        'mindate': mindate,
        'maxdate': maxdate,
        'minlat': str(minlat) if minlat is not None else None,
        'maxlat': str(maxlat) if maxlat is not None else None,
        'minlon': str(minlon) if minlon is not None else None,
        'maxlon': str(maxlon) if maxlon is not None else None
    })

    if 'timeout' not in kwargs:
        kwargs['timeout'] = 30

    url = "https://www.pangaea.de/advanced/search.php"
    logging.debug("Sending request to PANGAEA with parameters: %s", params)
    
    # Rate limiting - PANGAEA struggles with rapid requests
    import time
    time.sleep(5)  # 5s delay between requests - be gentle with PANGAEA
    
    response = requests.get(url, params=params, **kwargs)
    response.raise_for_status()
    logging.debug(f"URL: {response.url}")
    logging.debug(f"Response Status Code: {response.status_code}")
    results = response.json()
    logging.debug("Received response from PANGAEA")

    # Save the initial JSON response to transit.json
    try:
        transit_json_path = os.path.join(WorkspaceManager.get_sandbox_path(), 'transit.json')
        with open(transit_json_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Initial JSON response saved to {transit_json_path}")
    except Exception as e:
        logging.warning(f"Could not save transit.json: {e}")

    # --- PHASE 1: Parse basic results (fast) ---
    search_results = []
    for index, res in enumerate(results.get('results', [])):
        html_content = res.get('html', '')
        res['doi'] = f"https://doi.org/{res['URI'].replace('doi:', '')}"
        parsed_res = parse_res(html_content)
        res.update(parsed_res)

        search_results.append({
            'index': index,
            'doi': res['doi'],
            'name': res.get('citation', 'No name available'),
            'score': res.get('score', 0),
            'file_urls': parsed_res['file_urls']
        })

    # --- PHASE 2: Fetch metadata in PARALLEL using ThreadPoolExecutor (fast!) ---
    metadata_cache = {}
    dois_to_fetch = [r['doi'] for r in search_results]

    if dois_to_fetch:
        logging.info(f"Fetching metadata for {len(dois_to_fetch)} datasets in parallel...")

        def fetch_metadata_safe(doi):
            """Wrapper to catch exceptions and return None on failure"""
            try:
                return doi, fetch_metadata_fast(doi)
            except Exception as e:
                logging.debug(f"Fast fetch failed for {doi}: {e}")
                return doi, None

        # Use ThreadPoolExecutor for parallel fetching (max 8 workers to avoid rate limits)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_metadata_safe, doi) for doi in dois_to_fetch]

            for future in as_completed(futures):
                try:
                    doi, metadata = future.result()
                    metadata_cache[doi] = metadata
                except Exception as e:
                    logging.debug(f"Error in parallel fetch: {e}")

        logging.info(f"Parallel metadata fetch completed: {sum(1 for v in metadata_cache.values() if v)} successful")

    # --- PHASE 3: Combine results with MACHINE-ACTIONABLE metadata ---
    parsed = []
    for res in search_results:
        doi = res['doi']
        metadata = metadata_cache.get(doi)

        # Use fast metadata if available, otherwise fallback to sequential fetch
        if metadata:
            abstract = metadata.get('abstract') or "No description available"

            # --- MACHINE INTELLIGENCE: Add feature tags to description ---
            flags = metadata.get('feature_flags', {})
            matrix_stats = metadata.get('matrix_stats', {})
            feature_tags = format_feature_tags(flags, matrix_stats)

            # Prepend tags to abstract for agent routing
            if feature_tags:
                abstract = f"{feature_tags} {abstract}"

            # Use RICH parameters with stats if available
            params_rich = metadata.get('parameters_rich', [])
            if params_rich:
                parameters = format_parameters(params_rich, max_count=12)
            else:
                # Fallback to simple parameters
                params = metadata.get('parameters', [])
                parameters = format_parameters(params, max_count=10)
        else:
            # Fallback to slower pangaeapy for missing metadata
            logging.debug(f"Fallback to pangaeapy for {doi}")
            abstract, parameters = fetch_dataset_details(doi)

        short_description = " ".join(abstract.split()[:100]) + "..." if len(abstract.split()) > 100 else abstract

        parsed.append({
            'Number': res['index'] + 1,
            'Name': res['name'],
            'DOI': doi,
            'DOI Number': doi.split('/')[-1],
            'Description': abstract,
            'Short Description': short_description,
            'Score': res['score'],
            'Parameters': parameters,
            'file_urls': res['file_urls']
        })

    df = pd.DataFrame(parsed)
    total_hits = results.get('totalCount', 0)
    df.attrs['total'] = total_hits
    df.attrs['max_score'] = results.get('maxScore', None)
    return df


def direct_access_doi(doi: str, session_data: dict = None):
    """
    Directly access datasets by DOI without using the search function.
    Uses "Shopping Cart" strategy - ACCUMULATES datasets instead of replacing them.

    Args:
        doi: One or more DOIs separated by commas

    Returns:
        tuple: (datasets_info, prompt_text)
    """
    import pandas as pd
    import logging
    import re
    import pangaeapy.pandataset as pdataset

    if session_data is None:
        session_data = {}

    # Parse input to get list of DOIs
    dois = [d.strip() for d in doi.split(',')]
    logging.info(f"Processing direct DOI access for: {dois}")

    new_datasets_list = []
    existing_dois = set()

    # --- LOAD EXISTING DATASETS (Shopping Cart logic - keep previous results) ---
    if "datasets_info" in session_data and isinstance(session_data["datasets_info"], pd.DataFrame):
        if not session_data["datasets_info"].empty:
            # Get current DOIs to avoid duplicates
            existing_dois = set(session_data["datasets_info"]['DOI'].tolist())
            logging.info(f"Shopping Cart: Found {len(existing_dois)} existing datasets in workspace")
    # ---------------------------------------------------------------

    for i, curr_doi in enumerate(dois):
        try:
            # Normalize DOI format if needed
            original_doi = curr_doi  # Keep original for reference
            pangaea_id = None

            if not curr_doi.startswith(("http://", "https://")):
                # Check if it's a PANGAEA ID number
                if re.match(r"^\d+$", curr_doi):
                    pangaea_id = curr_doi
                    curr_doi = f"https://doi.pangaea.de/10.1594/PANGAEA.{curr_doi}"
                # Check if it's a PANGAEA DOI with PANGAEA prefix
                elif curr_doi.startswith("PANGAEA."):
                    pangaea_id = curr_doi.split("PANGAEA.")[1]
                    curr_doi = f"https://doi.pangaea.de/10.1594/{curr_doi}"
                # Check if it's just the DOI part
                elif curr_doi.startswith("10.1594/"):
                    match = re.search(r"PANGAEA\.(\d+)", curr_doi)
                    pangaea_id = match.group(1) if match else None
                    curr_doi = f"https://doi.org/{curr_doi}"
                else:
                    # Assume it's a PANGAEA ID
                    pangaea_id = curr_doi
                    curr_doi = f"https://doi.pangaea.de/10.1594/PANGAEA.{curr_doi}"
            else:
                # Extract ID from full URL
                match = re.search(r"PANGAEA\.(\d+)", curr_doi)
                pangaea_id = match.group(1) if match else None

            # --- SKIP IF ALREADY IN WORKSPACE (Shopping Cart deduplication) ---
            if curr_doi in existing_dois:
                logging.info(f"Shopping Cart: Skipping duplicate DOI: {curr_doi}")
                continue
            # -----------------------------------------------------------------

            logging.info(f"Fetching details for DOI: {curr_doi} (ID: {pangaea_id})")

            # Get dataset details directly
            try:
                if pangaea_id and pangaea_id.isdigit():
                    dataset = pdataset.PanDataSet(id=int(pangaea_id))
                    dataset.setMetadata()

                    # Get title
                    dataset_title = getattr(dataset, 'title', None) or f"Dataset from DOI: {curr_doi}"

                    # Get abstract
                    abstract = getattr(dataset, 'abstract', "No description available") or "No description available"

                    # Get parameters
                    param_dict = dataset.getParamDict()
                    short_names = param_dict.get('shortName', [])
                    parameters = ', '.join(short_names) + "..." if len(short_names) > 10 else ', '.join(short_names)
                else:
                    dataset_title = f"Dataset from DOI: {curr_doi}"
                    abstract = "No description available"
                    parameters = "No parameters available"
            except Exception as e:
                logging.error(f"Error getting dataset details for DOI {curr_doi}: {str(e)}")
                dataset_title = f"Dataset from DOI: {curr_doi}"
                abstract = f"Error fetching details: {str(e)}"
                parameters = "No parameters available due to error"

            short_description = " ".join(abstract.split()[:100]) + "..." if len(abstract.split()) > 100 else abstract

            # Add dataset to NEW list
            new_datasets_list.append({
                'Name': dataset_title,
                'DOI': curr_doi,
                'DOI Number': curr_doi.split('/')[-1] if '/' in curr_doi else curr_doi,
                'Description': abstract,
                'Short Description': short_description,
                'Score': 1.0,  # Default score
                'Parameters': parameters,
                'file_urls': []
            })
            existing_dois.add(curr_doi)

            logging.info(f"Shopping Cart: Added new DOI: {curr_doi} with title: {dataset_title}")

        except Exception as e:
            logging.error(f"Error processing DOI {curr_doi}: {str(e)}")

    # --- Handle case where no new datasets and no existing datasets ---
    if not new_datasets_list and not existing_dois:
        if "messages_search" in session_data:
            session_data["messages_search"].append({
                "role": "assistant",
                "content": "No valid datasets found from the provided DOIs."
            })
        # Return empty DataFrame (not None!) to avoid .empty / len() errors downstream
        return pd.DataFrame(), "No valid datasets found."

    # --- MERGE LOGIC (Shopping Cart accumulation) ---
    new_df = pd.DataFrame(new_datasets_list)

    if "datasets_info" in session_data and isinstance(session_data["datasets_info"], pd.DataFrame):
        if not session_data["datasets_info"].empty:
            # Add new datasets to existing ones
            combined_df = pd.concat([session_data["datasets_info"], new_df], ignore_index=True)
            # Deduplicate by DOI (keep last occurrence in case of updates)
            datasets_info = combined_df.drop_duplicates(subset=['DOI'], keep='last').reset_index(drop=True)
            logging.info(f"Shopping Cart: Merged {len(new_datasets_list)} new + {len(session_data['datasets_info'])} existing = {len(datasets_info)} total")
        else:
            datasets_info = new_df
    else:
        datasets_info = new_df

    # Renumber sequentially (1, 2, 3...)
    datasets_info['Number'] = datasets_info.index + 1

    # Store in session state
    session_data["datasets_info"] = datasets_info

    # --- NOTIFICATION MESSAGE (Shopping Cart style) ---
    # Only notify if something new was added
    if new_datasets_list:
        msg = f"**Shopping Cart Updated:** Added {len(new_datasets_list)} new datasets. Total in workspace: {len(datasets_info)}."
        if "messages_search" in session_data:
            session_data["messages_search"].append({
                "role": "assistant",
                "content": msg,
                "table": datasets_info.to_json(orient="split")
            })

    # Create prompt for the search agent with ALL accumulated datasets
    datasets_description = ""
    for i, row in datasets_info.iterrows():
        datasets_description += (
            f"Dataset {i + 1}: {row['Name']} ({row['Parameters']})\n"
        )

    prompt_search = (
        f"Total accumulated datasets in workspace: {len(datasets_info)}.\n"
        f"Available datasets:\n{datasets_description}\n"
        "These datasets are accumulated in the workspace. You can perform analysis across all of them."
    )

    return datasets_info, prompt_search
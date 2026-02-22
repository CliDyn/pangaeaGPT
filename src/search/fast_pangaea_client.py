# src/search/fast_pangaea_client.py
"""
Fast OAI-PMH Client for PANGAEA Metadata
Replaces slow pangaeapy calls with direct XML parsing for search results.
"""

import logging
import re
import requests
from typing import Dict, Any, Optional, List
from xml.etree import ElementTree as ET

# OAI-PMH XML Namespaces
NS = {
    'oai': 'http://www.openarchives.org/OAI/2.0/',
    'md': 'http://www.pangaea.de/MetaData'
}

BASE_URL = "https://ws.pangaea.de/oai/provider"
REQUEST_TIMEOUT = 15  # seconds


def extract_pangaea_id(doi: str) -> Optional[str]:
    """
    Extract PANGAEA numeric ID from various DOI formats.

    Supported formats:
    - https://doi.org/10.1594/PANGAEA.123456
    - https://doi.pangaea.de/10.1594/PANGAEA.123456
    - 10.1594/PANGAEA.123456
    - PANGAEA.123456
    - 123456
    """
    if not doi:
        return None

    # Try to extract numeric ID
    match = re.search(r'PANGAEA[.\s]*(\d+)', doi, re.IGNORECASE)
    if match:
        return match.group(1)

    # Check if it's just a number
    if re.match(r'^\d+$', doi.strip()):
        return doi.strip()

    # Try to extract from DOI path
    match = re.search(r'/(\d+)/?$', doi)
    if match:
        return match.group(1)

    return None


def fetch_metadata_fast(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch PANGAEA metadata using OAI-PMH protocol (pan_md format).
    Much faster than pangaeapy for metadata-only retrieval.

    Args:
        doi: PANGAEA DOI in any supported format

    Returns:
        Dict with keys: title, abstract, parameters, authors, year
        None if fetch fails
    """
    pangaea_id = extract_pangaea_id(doi)
    if not pangaea_id:
        logging.warning(f"Could not extract PANGAEA ID from: {doi}")
        return None

    # Build OAI-PMH GetRecord URL
    identifier = f"oai:pangaea.de:doi:10.1594/PANGAEA.{pangaea_id}"
    params = {
        'verb': 'GetRecord',
        'identifier': identifier,
        'metadataPrefix': 'pan_md'
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.content)

        # Check for errors
        error = root.find('.//oai:error', NS)
        if error is not None:
            logging.warning(f"OAI-PMH error for {pangaea_id}: {error.text}")
            return None

        # Find metadata record
        record = root.find('.//oai:record', NS)
        if record is None:
            logging.warning(f"No record found for {pangaea_id}")
            return None

        metadata = record.find('.//oai:metadata', NS)
        if metadata is None:
            logging.warning(f"No metadata found for {pangaea_id}")
            return None

        # Extract fields from pan_md format
        result = {
            'pangaea_id': pangaea_id,
            'title': None,
            'abstract': None,
            'parameters': [],
            'authors': [],
            'year': None
        }

        # Title
        title_elem = metadata.find('.//md:citation/md:title', NS)
        if title_elem is not None and title_elem.text:
            result['title'] = title_elem.text.strip()

        # Abstract
        abstract_elem = metadata.find('.//md:abstract', NS)
        if abstract_elem is not None and abstract_elem.text:
            result['abstract'] = abstract_elem.text.strip()

        # Year
        year_elem = metadata.find('.//md:citation/md:year', NS)
        if year_elem is not None and year_elem.text:
            result['year'] = year_elem.text.strip()

        # Authors
        for author in metadata.findall('.//md:citation/md:author', NS):
            last_name = author.find('md:lastName', NS)
            first_name = author.find('md:firstName', NS)
            if last_name is not None and last_name.text:
                name = last_name.text
                if first_name is not None and first_name.text:
                    name = f"{last_name.text}, {first_name.text}"
                result['authors'].append(name)

        # Parameters (variable names)
        for param in metadata.findall('.//md:matrixColumn', NS):
            param_elem = param.find('md:parameter/md:name', NS)
            if param_elem is not None and param_elem.text:
                result['parameters'].append(param_elem.text.strip())
            else:
                # Try shortName as fallback
                short_name = param.find('md:parameter/md:shortName', NS)
                if short_name is not None and short_name.text:
                    result['parameters'].append(short_name.text.strip())

        # Remove duplicates from parameters while preserving order
        seen = set()
        unique_params = []
        for p in result['parameters']:
            if p not in seen:
                seen.add(p)
                unique_params.append(p)
        result['parameters'] = unique_params

        return result

    except requests.exceptions.Timeout:
        logging.warning(f"Timeout fetching metadata for {pangaea_id}")
        return None
    except requests.exceptions.RequestException as e:
        logging.warning(f"Request error for {pangaea_id}: {e}")
        return None
    except ET.ParseError as e:
        logging.warning(f"XML parse error for {pangaea_id}: {e}")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error fetching metadata for {pangaea_id}: {e}")
        return None


def format_parameters(params: List[str], max_count: int = 10) -> str:
    """
    Format parameters list as a comma-separated string.
    Truncates with '...' if more than max_count.
    """
    if not params:
        return "No parameters available"

    if len(params) > max_count:
        return ', '.join(params[:max_count]) + "..."
    return ', '.join(params)


def fetch_metadata_batch(dois: List[str], max_workers: int = 5) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetch metadata for multiple DOIs in parallel using ThreadPoolExecutor.

    Args:
        dois: List of DOI strings
        max_workers: Maximum parallel requests (default 5 to avoid rate limiting)

    Returns:
        Dict mapping DOI -> metadata dict (or None if failed)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_doi = {
            executor.submit(fetch_metadata_fast, doi): doi
            for doi in dois
        }

        # Collect results as they complete
        for future in as_completed(future_to_doi):
            doi = future_to_doi[future]
            try:
                result = future.result()
                results[doi] = result
            except Exception as e:
                logging.warning(f"Error fetching {doi}: {e}")
                results[doi] = None

            # Small delay between completions to be rate-limit friendly
            time.sleep(0.1)

    return results

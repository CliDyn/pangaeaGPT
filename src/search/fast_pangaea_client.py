# src/search/fast_pangaea_client.py
"""
Fast OAI-PMH Client for PANGAEA Metadata
With Machine-Actionable Intelligence for AI Agents.

Features:
- Matrix dimensions (row/col counts) for memory safety
- Column statistics (min/max) for instant range queries
- Feature flags for smart agent routing
- Semantic IDs for parameter disambiguation
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def txt(elem) -> Optional[str]:
    """Safe text extraction from XML element"""
    return elem.text.strip() if elem is not None and elem.text else None


def attr(elem, name) -> Optional[str]:
    """Safe attribute extraction from XML element"""
    return elem.get(name) if elem is not None else None


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


# =============================================================================
# RICH PARAMETER PARSING (with Stats & Semantics)
# =============================================================================

def parse_matrix_column(col_elem) -> Optional[dict]:
    """
    Parse ColumnType with FULL intelligence:
    - Parameter name/shortName/unit
    - Semantic ID for disambiguation
    - Pre-calculated statistics (min/max/mean)
    - Column type and format
    """
    if col_elem is None:
        return None

    param_elem = col_elem.find('md:parameter', NS)
    if param_elem is None:
        return None

    # Get semantic term (URI to standard vocabulary)
    term_elem = param_elem.find('md:term', NS)
    term_uri = attr(term_elem, 'URI') if term_elem is not None else None

    # Extract pre-calculated statistics (INSTANT AI ANSWERS!)
    stats = {}
    for stat_name in ['min', 'max', 'mean', 'median']:
        val = txt(col_elem.find(f'md:{stat_name}', NS))
        if val:
            stats[stat_name] = val

    return {
        'name': txt(param_elem.find('md:name', NS)),
        'shortName': txt(param_elem.find('md:shortName', NS)),
        'unit': txt(param_elem.find('md:unit', NS)),
        'group': txt(param_elem.find('md:group', NS)),
        'id': attr(param_elem, 'id'),           # Semantic ID (e.g., "2001" for Temp)
        'semantic_uri': term_uri,               # Link to vocabulary
        'type': attr(col_elem, 'type'),         # geocode, data, etc.
        'format': attr(col_elem, 'format'),     # Date format, etc.
        'source': attr(col_elem, 'source'),
        'stats': stats if stats else None       # Pre-calculated ranges!
    }


def parse_distribution(md_root) -> List[dict]:
    """Extract direct download links (the "Golden Link")"""
    dists = []
    for dist in md_root.findall('md:distribution', NS):
        res = dist.find('md:onlineResource', NS)
        if res is not None:
            dists.append({
                'uri': txt(res.find('md:URI', NS)),
                'function': attr(res, 'function'),  # e.g., "download"
                'format': attr(dist, 'format')      # e.g., "text/tab-separated-values"
            })
    return dists


def compute_feature_flags(matrix_stats: dict, parameters: List[dict]) -> dict:
    """
    Compute AI-ready feature flags for smart agent routing.

    Returns flags like:
    - is_heavy: Dataset has >1M rows (use Dask!)
    - likely_map: Has lat/lon (send to VisualizationAgent)
    - likely_profile: Has depth but no time (vertical profile)
    - has_time: Time-series data
    - has_depth: Depth/pressure data
    - has_locations: Geocoded data
    """
    flags = {
        'is_heavy': False,
        'likely_map': False,
        'likely_profile': False,
        'has_time': False,
        'has_depth': False,
        'has_locations': False,
        'is_parent': False
    }

    # Check row count for memory safety
    row_count = matrix_stats.get('rowCount')
    if row_count:
        try:
            if int(row_count) > 1_000_000:
                flags['is_heavy'] = True
        except ValueError:
            pass

    # Parent datasets have no data columns
    if not parameters:
        flags['is_parent'] = True
        return flags

    # Analyze parameters for context
    for p in parameters:
        if not p:
            continue

        p_name = (p.get('name') or '').lower()
        p_short = (p.get('shortName') or '').lower()
        p_source = (p.get('source') or '').lower()
        p_type = (p.get('type') or '').lower()
        combined = f"{p_name} {p_short}"

        # Spatial detection
        if 'latitude' in combined or 'longitude' in combined or p_source == 'geocode':
            flags['has_locations'] = True
            flags['likely_map'] = True

        # Depth detection
        if 'depth' in combined or 'pressure' in combined:
            flags['has_depth'] = True

        # Time detection
        if 'date' in combined or 'time' in combined or p_type == 'datetime':
            flags['has_time'] = True

    # Heuristics for profile detection
    if flags['has_depth'] and not flags['has_time']:
        flags['likely_profile'] = True

    return flags


# =============================================================================
# MAIN FETCH FUNCTION
# =============================================================================

def fetch_metadata_fast(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch PANGAEA metadata with FULL machine-actionable intelligence.

    Returns dict with:
    - title, abstract, year, authors (human-readable)
    - parameters_rich: List of parameter dicts with stats
    - matrix_stats: {rowCount, colCount} for memory planning
    - feature_flags: {is_heavy, likely_map, ...} for routing
    - distributions: Direct download links
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

        # Find MetaData root element
        md_root = metadata.find('md:MetaData', NS)
        if md_root is None:
            logging.warning(f"No MetaData element for {pangaea_id}")
            return None

        # === 1. MATRIX STRUCTURE (Memory Safety) ===
        matrix_elem = md_root.find('md:matrix', NS)
        matrix_stats = {}
        if matrix_elem is not None:
            matrix_stats = {
                'colCount': attr(matrix_elem, 'colCount'),
                'rowCount': attr(matrix_elem, 'rowCount')  # Critical for Dask vs Pandas!
            }

        # === 2. RICH PARAMETERS (with Stats) ===
        parameters_rich = []
        parameters_simple = []  # For backward compatibility

        for col in md_root.findall('md:matrixColumn', NS):
            parsed = parse_matrix_column(col)
            if parsed:
                parameters_rich.append(parsed)
                # Also build simple list for backward compatibility
                name = parsed.get('shortName') or parsed.get('name')
                if name:
                    parameters_simple.append(name)

        # Remove duplicates from simple list
        seen = set()
        unique_params = []
        for p in parameters_simple:
            if p not in seen:
                seen.add(p)
                unique_params.append(p)

        # === 3. FEATURE FLAGS (Agent Routing) ===
        feature_flags = compute_feature_flags(matrix_stats, parameters_rich)

        # === 4. DISTRIBUTIONS (Direct Links) ===
        distributions = parse_distribution(md_root)

        # === 5. BASIC INFO ===
        title = txt(md_root.find('md:citation/md:title', NS))
        abstract = txt(md_root.find('md:abstract', NS))
        year = txt(md_root.find('md:citation/md:year', NS))

        # Authors
        authors = []
        for author in md_root.findall('md:citation/md:author', NS):
            last_name = txt(author.find('md:lastName', NS))
            first_name = txt(author.find('md:firstName', NS))
            if last_name:
                name = f"{last_name}, {first_name}" if first_name else last_name
                authors.append(name)

        return {
            'pangaea_id': pangaea_id,
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,

            # Machine-Actionable Intelligence
            'parameters': unique_params,        # Simple list (backward compat)
            'parameters_rich': parameters_rich, # Full dicts with stats!
            'matrix_stats': matrix_stats,       # Row/col counts
            'feature_flags': feature_flags,     # Agent routing hints
            'distributions': distributions      # Direct download links
        }

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


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_parameters(params: List[Any], max_count: int = 10) -> str:
    """
    Format parameters with ranges for AI Agents.

    Handles both:
    - Simple string list (backward compat)
    - Rich parameter dicts with stats (shows ranges!)

    Output example: "Temp[-1.5..4.0], Salinity[32..35], Depth"
    """
    if not params:
        return "No parameters available"

    out_list = []

    # Check if we have rich parameter dicts
    if params and isinstance(params[0], dict):
        for p in params[:max_count]:
            name = p.get('shortName') or p.get('name')
            if not name:
                continue

            stats = p.get('stats') or {}
            unit = p.get('unit')

            # Build rich string with stats
            if stats and stats.get('min') is not None and stats.get('max') is not None:
                # Show range: "Temp[-1.5..4.0]"
                out_list.append(f"{name}[{stats['min']}..{stats['max']}]")
            elif unit:
                # Show unit: "Temp (°C)"
                out_list.append(f"{name} ({unit})")
            else:
                out_list.append(name)
    else:
        # Simple string list (backward compatibility)
        out_list = [str(p) for p in params[:max_count] if p]

    if not out_list:
        return "No parameters available"

    result = ', '.join(out_list)
    if len(params) > max_count:
        result += "..."

    # Truncate if too long
    if len(result) > 300:
        result = result[:300] + "..."

    return result


def format_feature_tags(flags: dict, matrix_stats: dict) -> str:
    """
    Format feature flags as tags for agent prompts.

    Output example: "[MAP] [HEAVY_DATA] [Rows:5000000]"
    """
    tags = []

    if flags.get('likely_map'):
        tags.append("MAP")
    if flags.get('likely_profile'):
        tags.append("PROFILE")
    if flags.get('is_heavy'):
        tags.append("HEAVY_DATA")
    if flags.get('is_parent'):
        tags.append("COLLECTION")
    if flags.get('has_time'):
        tags.append("TIMESERIES")

    # Add row count if available
    if matrix_stats.get('rowCount'):
        tags.append(f"Rows:{matrix_stats['rowCount']}")

    return f"[{' '.join(tags)}]" if tags else ""


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

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
        future_to_doi = {
            executor.submit(fetch_metadata_fast, doi): doi
            for doi in dois
        }

        for future in as_completed(future_to_doi):
            doi = future_to_doi[future]
            try:
                result = future.result()
                results[doi] = result
            except Exception as e:
                logging.warning(f"Error fetching {doi}: {e}")
                results[doi] = None

            time.sleep(0.1)

    return results

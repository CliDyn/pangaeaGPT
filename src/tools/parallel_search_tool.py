# src/tools/parallel_search_tool.py
"""
Enhanced Sequential PANGAEA Search Tool with Full Metadata Preservation
NOW WITH RATE LIMIT PROTECTION - NO MORE KICKS!
"""

import logging
import time
import streamlit as st
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
import pandas as pd

from ..search.search_pg_default import pg_search_default


class ParallelSearchArgs(BaseModel):
    search_queries: List[str] = Field(description="List of SEARCH QUERY STRINGS (NOT DOIs, NOT full text) to execute sequentially. Example: ['temperature data', 'salinity measurements', 'arctic ocean CTD']")
    count_per_query: Optional[int] = Field(default=30, description="Number of results per query (5-50)")
    mindate: Optional[str] = Field(default=None, description="Minimum date in 'YYYY-MM-DD' format")
    maxdate: Optional[str] = Field(default=None, description="Maximum date in 'YYYY-MM-DD' format")
    minlat: Optional[float] = Field(default=None, description="Minimum latitude in decimal degrees")
    maxlat: Optional[float] = Field(default=None, description="Maximum latitude in decimal degrees")
    minlon: Optional[float] = Field(default=None, description="Minimum longitude in decimal degrees")
    maxlon: Optional[float] = Field(default=None, description="Maximum longitude in decimal degrees")


@dataclass
class SearchResult:
    """Container for search results with full metadata preservation"""
    query: str
    datasets_df: pd.DataFrame
    execution_time: float
    success: bool
    total_found: int = 0
    error: Optional[str] = None


class EnhancedSequentialSearchExecutor:
    """
    Enhanced SEQUENTIAL search executor with RATE LIMIT PROTECTION.
    NOW WITH DELAYS TO PREVENT GETTING KICKED!
    """
    
    def __init__(self, delay_between_searches: float = 20.0):
        self.delay_between_searches = delay_between_searches  # 3 seconds delay by default
        
    def execute_single_search_with_metadata(self, query: str, count: int = 30, **search_params) -> SearchResult:
        """
        Execute a single search query with FULL metadata extraction.
        Returns the complete DataFrame with all metadata fields preserved.
        """
        start_time = time.time()
        
        try:
            logging.info(f"🔍 Executing search: '{query}' (count={count})")
            
            # Execute the search using the existing function that extracts ALL metadata
            datasets_df = pg_search_default(
                query, 
                count=count,
                mindate=search_params.get('mindate'),
                maxdate=search_params.get('maxdate'),
                minlat=search_params.get('minlat'),
                maxlat=search_params.get('maxlat'),
                minlon=search_params.get('minlon'),
                maxlon=search_params.get('maxlon')
            )
            
            execution_time = time.time() - start_time
            total_found = datasets_df.attrs.get('total', 0) if hasattr(datasets_df, 'attrs') else len(datasets_df)
            
            logging.info(f"✅ Search '{query}' completed: {len(datasets_df)} datasets found in {execution_time:.2f}s")
            
            return SearchResult(
                query=query,
                datasets_df=datasets_df,
                execution_time=execution_time,
                success=True,
                total_found=total_found
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"❌ Search '{query}' failed: {str(e)}")
            
            return SearchResult(
                query=query,
                datasets_df=pd.DataFrame(),
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    def execute_sequential_searches_with_metadata(self, queries: List[str], count_per_query: int = 30, 
                                                 **search_params) -> List[SearchResult]:
        """
        Execute multiple searches SEQUENTIALLY with DELAYS, preserving ALL metadata from each search.
        NO MORE PARALLEL EXECUTION - RATE LIMIT SAFE!
        """
        if not queries:
            return []
        
        logging.info(f"🚀 Starting SEQUENTIAL execution of {len(queries)} searches with {self.delay_between_searches}s delay between each")
        
        # Initialize progress tracking
        if hasattr(st, 'session_state') and 'parallel_search_progress' in st.session_state:
            st.session_state.parallel_search_progress.update({
                'total_queries': len(queries),
                'completed_queries': 0,
                'current_query': '',
                'metadata_phase': True,
                'results': []
            })
        
        search_results = []
        
        # SEQUENTIAL EXECUTION WITH DELAYS
        for i, query in enumerate(queries):
            logging.info(f"📍 Processing query {i+1}/{len(queries)}: '{query}'")
            
            # Update progress
            if hasattr(st, 'session_state') and 'parallel_search_progress' in st.session_state:
                st.session_state.parallel_search_progress['current_query'] = query
            
            # Execute the search
            result = self.execute_single_search_with_metadata(
                query, count_per_query, **search_params
            )
            
            search_results.append(result)
            
            # Update progress tracking
            if hasattr(st, 'session_state') and 'parallel_search_progress' in st.session_state:
                st.session_state.parallel_search_progress['completed_queries'] = i + 1
                
                # Add result summary for final display
                result_summary = {
                    'query': query,
                    'success': result.success,
                    'count': len(result.datasets_df),
                    'execution_time': result.execution_time
                }
                st.session_state.parallel_search_progress['results'].append(result_summary)
            
            if result.success:
                logging.info(f"📊 Query '{query}': {len(result.datasets_df)} datasets with metadata extracted")
            else:
                logging.warning(f"⚠️ Query '{query}' failed: {result.error}")
            
            # CRITICAL: ADD DELAY BETWEEN SEARCHES (except after the last one)
            if i < len(queries) - 1:
                logging.info(f"⏳ Waiting {self.delay_between_searches}s before next search to avoid rate limits...")
                time.sleep(self.delay_between_searches)
        
        logging.info(f"🏁 Sequential search completed: {sum(1 for r in search_results if r.success)}/{len(queries)} successful")
        return search_results


def parallel_search_pangaea(search_queries: List[str], count_per_query: int = 30,
                          mindate: Optional[str] = None, maxdate: Optional[str] = None,
                          minlat: Optional[float] = None, maxlat: Optional[float] = None,
                          minlon: Optional[float] = None, maxlon: Optional[float] = None) -> Dict[str, Any]:
    """
    Enhanced SEQUENTIAL search function that executes multiple PANGAEA searches ONE BY ONE
    with DELAYS to prevent rate limiting, while preserving ALL metadata for intelligent consolidation.
    
    Returns:
        Dict containing:
        - all_datasets: List of dataset dicts with full metadata (Score, Name, Parameters, etc.)
        - all_dois: Simple DOI list for backward compatibility
        - search_results: Detailed results per query
        - execution_stats: Performance and quality metrics
    """
    
    if not search_queries:
        return {
            'all_datasets': [],
            'all_dois': [],
            'search_results': [],
            'execution_stats': {
                'total_queries': 0,
                'successful_queries': 0,
                'total_datasets': 0,
                'unique_dois': 0,
                'total_execution_time': 0,
                'avg_score': 0
            },
            'message': 'No search queries provided'
        }
    
    # Handle single query case efficiently
    if len(search_queries) == 1:
        executor = EnhancedSequentialSearchExecutor(delay_between_searches=0)  # No delay for single query
        result = executor.execute_single_search_with_metadata(
            search_queries[0], count_per_query,
            mindate=mindate, maxdate=maxdate,
            minlat=minlat, maxlat=maxlat, 
            minlon=minlon, maxlon=maxlon
        )
        
        # Convert DataFrame to dataset list with metadata
        all_datasets = []
        all_dois = []
        
        if result.success and not result.datasets_df.empty:
            for _, row in result.datasets_df.iterrows():
                dataset_dict = {
                    'DOI': row.get('DOI', ''),
                    'Score': row.get('Score', 0),
                    'Name': row.get('Name', ''),
                    'Description': row.get('Description', ''),
                    'Short_Description': row.get('Short Description', ''),
                    'Parameters': row.get('Parameters', ''),
                    'DOI_Number': row.get('DOI Number', ''),
                    'Number': row.get('Number', 0),
                    'search_query': result.query,
                    'file_urls': row.get('file_urls', [])
                }
                all_datasets.append(dataset_dict)
                all_dois.append(row.get('DOI', ''))
        
        return {
            'all_datasets': all_datasets,
            'all_dois': all_dois,
            'search_results': [{
                'query': result.query,
                'success': result.success,
                'count': len(result.datasets_df),
                'execution_time': result.execution_time,
                'total_found': result.total_found,
                'error': result.error
            }],
            'execution_stats': {
                'total_queries': 1,
                'successful_queries': 1 if result.success else 0,
                'total_datasets': len(all_datasets),
                'unique_dois': len(set(all_dois)),
                'total_execution_time': result.execution_time,
                'avg_score': sum(d['Score'] for d in all_datasets) / len(all_datasets) if all_datasets else 0
            },
            'message': f"Single search completed: {len(all_datasets)} datasets found with metadata"
        }
    
    # Multiple queries - use SEQUENTIAL execution with DELAYS
    start_time = time.time()
    executor = EnhancedSequentialSearchExecutor(delay_between_searches=3.0)  # 3 second delay between searches
    
    # Limit to maximum 5 queries to prevent excessive requests
    if len(search_queries) > 5:
        logging.warning(f"⚠️ Limiting search to first 5 queries (from {len(search_queries)}) to prevent rate limiting")
        search_queries = search_queries[:5]
    
    search_results = executor.execute_sequential_searches_with_metadata(
        search_queries, count_per_query,
        mindate=mindate, maxdate=maxdate,
        minlat=minlat, maxlat=maxlat,
        minlon=minlon, maxlon=maxlon
    )
    
    # Aggregate all datasets with full metadata preservation
    all_datasets = []
    all_dois = set()  # Use set for automatic deduplication
    query_summaries = []
    
    for result in search_results:
        query_summary = {
            'query': result.query,
            'success': result.success,
            'count': len(result.datasets_df),
            'execution_time': result.execution_time,
            'total_found': result.total_found,
            'error': result.error if not result.success else None
        }
        query_summaries.append(query_summary)
        
        if result.success and not result.datasets_df.empty:
            # Extract all metadata from each dataset
            for _, row in result.datasets_df.iterrows():
                dataset_dict = {
                    'DOI': row.get('DOI', ''),
                    'Score': row.get('Score', 0),
                    'Name': row.get('Name', ''),
                    'Description': row.get('Description', ''),
                    'Short_Description': row.get('Short Description', ''),
                    'Parameters': row.get('Parameters', ''),
                    'DOI_Number': row.get('DOI Number', ''),
                    'Number': row.get('Number', 0),
                    'search_query': result.query,  # Track which query found this dataset
                    'file_urls': row.get('file_urls', [])
                }
                all_datasets.append(dataset_dict)
                all_dois.add(row.get('DOI', ''))
    
    # Sort datasets by score (highest first) for intelligent selection
    all_datasets.sort(key=lambda x: x['Score'], reverse=True)
    
    # Calculate comprehensive execution statistics
    total_execution_time = time.time() - start_time
    successful_queries = sum(1 for r in search_results if r.success)
    avg_score = sum(d['Score'] for d in all_datasets) / len(all_datasets) if all_datasets else 0
    
    execution_stats = {
        'total_queries': len(search_queries),
        'successful_queries': successful_queries,
        'total_datasets': len(all_datasets),
        'unique_dois': len(all_dois),
        'total_execution_time': total_execution_time,
        'avg_execution_time_per_query': total_execution_time / len(search_queries),
        'search_mode': 'SEQUENTIAL (Rate-limit safe)',
        'delay_used': '3 seconds between searches',
        'avg_score': avg_score,
        'top_score': all_datasets[0]['Score'] if all_datasets else 0,
        'score_range': f"{min(d['Score'] for d in all_datasets):.1f}-{max(d['Score'] for d in all_datasets):.1f}" if all_datasets else "N/A"
    }
    
    logging.info(
        f"🎯 Sequential search completed: {successful_queries}/{len(search_queries)} successful, "
        f"{len(all_dois)} unique DOIs, avg score: {avg_score:.1f}, "
        f"time: {total_execution_time:.1f}s (with rate limit protection)"
    )
    
    return {
        'all_datasets': all_datasets,  # 🔥 FULL METADATA for intelligent consolidation!
        'all_dois': list(all_dois),    # Simple DOI list for backward compatibility
        'search_results': query_summaries,
        'execution_stats': execution_stats,
        'message': f"🛡️ Rate-limit safe sequential search completed: {successful_queries}/{len(search_queries)} queries successful, "
                  f"{len(all_dois)} unique DOIs with full metadata extracted "
                  f"(avg score: {avg_score:.1f}, with 3s delays between searches)"
    }
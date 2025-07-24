# src/tools/parallel_search_tool.py
"""
Enhanced Parallel PANGAEA Search Tool with Full Metadata Preservation
NOW WITH INTELLIGENT METADATA EXTRACTION AND CONSOLIDATION!
"""

import logging
import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
import pandas as pd

from ..search.search_pg_default import pg_search_default


class ParallelSearchArgs(BaseModel):
    search_queries: List[str] = Field(description="List of SEARCH QUERY STRINGS (NOT DOIs, NOT full text) to execute in parallel. Example: ['temperature data', 'salinity measurements', 'arctic ocean CTD']")
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


class EnhancedParallelSearchExecutor:
    """
    Enhanced parallel search executor that preserves ALL metadata during parallel execution.
    NOW WITH INTELLIGENT METADATA-BASED CONSOLIDATION!
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = min(max_workers, 5)  # Limit to prevent overwhelming PANGAEA
        
    def execute_single_search_with_metadata(self, query: str, count: int = 30, **search_params) -> SearchResult:
        """
        Execute a single search query with FULL metadata extraction.
        Returns the complete DataFrame with all metadata fields preserved.
        """
        start_time = time.time()
        
        try:
            logging.info(f"🔍 Executing parallel search: '{query}' (count={count})")
            
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
    
    def execute_parallel_searches_with_metadata(self, queries: List[str], count_per_query: int = 30, 
                                              **search_params) -> List[SearchResult]:
        """
        Execute multiple searches in parallel, preserving ALL metadata from each search.
        """
        if not queries:
            return []
        
        logging.info(f"🚀 Starting parallel execution of {len(queries)} searches with metadata preservation")
        
        # Initialize progress tracking (no threading needed)
        if hasattr(st, 'session_state') and 'parallel_search_progress' in st.session_state:
            st.session_state.parallel_search_progress.update({
                'total_queries': len(queries),
                'completed_queries': 0,
                'current_query': '',
                'metadata_phase': True,
                'results': []
            })
        
        search_results = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all searches
            future_to_query = {
                executor.submit(
                    self.execute_single_search_with_metadata, 
                    query, count_per_query, **search_params
                ): query
                for query in queries
            }
            
            # Process completed searches as they finish
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                
                try:
                    result = future.result()
                    search_results.append(result)
                    
                    # Simple progress update (no threading conflicts)
                    if hasattr(st, 'session_state') and 'parallel_search_progress' in st.session_state:
                        st.session_state.parallel_search_progress['completed_queries'] += 1
                        st.session_state.parallel_search_progress['current_query'] = query
                        
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
                        
                except Exception as e:
                    logging.error(f"❌ Error processing result for '{query}': {str(e)}")
                    search_results.append(SearchResult(
                        query=query,
                        datasets_df=pd.DataFrame(),
                        execution_time=0,
                        success=False,
                        error=str(e)
                    ))
        
        logging.info(f"🏁 Parallel search completed: {sum(1 for r in search_results if r.success)}/{len(queries)} successful")
        return search_results


def parallel_search_pangaea(search_queries: List[str], count_per_query: int = 30,
                          mindate: Optional[str] = None, maxdate: Optional[str] = None,
                          minlat: Optional[float] = None, maxlat: Optional[float] = None,
                          minlon: Optional[float] = None, maxlon: Optional[float] = None) -> Dict[str, Any]:
    """
    Enhanced parallel search function that executes multiple PANGAEA searches concurrently
    while preserving ALL metadata for intelligent consolidation.
    
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
        executor = EnhancedParallelSearchExecutor(max_workers=1)
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
    
    # Multiple queries - use parallel execution with metadata preservation
    start_time = time.time()
    executor = EnhancedParallelSearchExecutor(max_workers=4)
    
    search_results = executor.execute_parallel_searches_with_metadata(
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
    
    # Estimate speedup (approximate)
    individual_times = [r.execution_time for r in search_results]
    sequential_estimate = sum(individual_times)
    speedup = sequential_estimate / max(total_execution_time, 0.1)
    
    execution_stats = {
        'total_queries': len(search_queries),
        'successful_queries': successful_queries,
        'total_datasets': len(all_datasets),
        'unique_dois': len(all_dois),
        'total_execution_time': total_execution_time,
        'avg_execution_time_per_query': total_execution_time / len(search_queries),
        'estimated_speedup': f"{speedup:.1f}x",
        'avg_score': avg_score,
        'top_score': all_datasets[0]['Score'] if all_datasets else 0,
        'score_range': f"{min(d['Score'] for d in all_datasets):.1f}-{max(d['Score'] for d in all_datasets):.1f}" if all_datasets else "N/A"
    }
    
    logging.info(
        f"🎯 Enhanced parallel search completed: {successful_queries}/{len(search_queries)} successful, "
        f"{len(all_dois)} unique DOIs, avg score: {avg_score:.1f}, "
        f"time: {total_execution_time:.1f}s ({speedup:.1f}x speedup)"
    )
    
    return {
        'all_datasets': all_datasets,  # 🔥 FULL METADATA for intelligent consolidation!
        'all_dois': list(all_dois),    # Simple DOI list for backward compatibility
        'search_results': query_summaries,
        'execution_stats': execution_stats,
        'message': f"🚀 Enhanced parallel search completed: {successful_queries}/{len(search_queries)} queries successful, "
                  f"{len(all_dois)} unique DOIs with full metadata extracted "
                  f"(avg score: {avg_score:.1f}, {speedup:.1f}x speedup)"
    }
# src/ui/directory_explorer.py
"""
Directory Explorer UI Component for PANGAEA GPT
Displays directory contents for plotting data and sandbox directories
"""

import os
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_directory_tree(path: str, max_depth: int = 5, current_depth: int = 0) -> Dict:
    """
    Recursively build a directory tree structure.
    
    Args:
        path: Root path to scan
        max_depth: Maximum depth to traverse
        current_depth: Current recursion depth
        
    Returns:
        dict: Tree structure with folders and files
    """
    tree = {
        "name": os.path.basename(path) or path,
        "path": path,
        "type": "directory",
        "children": [],
        "size": 0
    }
    
    if current_depth >= max_depth:
        return tree
    
    try:
        items = sorted(os.listdir(path))
        
        for item in items:
            # Always skip hidden files
            if item.startswith('.'):
                continue
                
            item_path = os.path.join(path, item)
            
            try:
                if os.path.isdir(item_path):
                    # Recursively get subdirectory
                    subtree = get_directory_tree(
                        item_path, 
                        max_depth, 
                        current_depth + 1
                    )
                    tree["children"].append(subtree)
                else:
                    # Add file info
                    size = os.path.getsize(item_path)
                    tree["children"].append({
                        "name": item,
                        "path": item_path,
                        "type": "file",
                        "size": size,
                        "modified": datetime.fromtimestamp(os.path.getmtime(item_path)).strftime("%Y-%m-%d %H:%M:%S")
                    })
                    tree["size"] += size
            except (PermissionError, OSError) as e:
                # Handle permission errors gracefully
                tree["children"].append({
                    "name": f"{item} (access denied)",
                    "path": item_path,
                    "type": "error",
                    "error": str(e)
                })
                
    except (PermissionError, OSError) as e:
        tree["error"] = str(e)
        
    return tree


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_file_icon(filename: str) -> str:
    """Get appropriate emoji icon based on file extension."""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    icon_map = {
        # Data files
        'csv': '📊',
        'xlsx': '📊',
        'xls': '📊',
        'json': '📋',
        'xml': '📋',
        'txt': '📄',
        'log': '📄',
        
        # Scientific data
        'nc': '🌍',
        'netcdf': '🌍',
        'cdf': '🌍',
        'hdf': '🗄️',
        'h5': '🗄️',
        'zarr': '🗄️',
        
        # Images
        'png': '🖼️',
        'jpg': '🖼️',
        'jpeg': '🖼️',
        'gif': '🖼️',
        'svg': '🖼️',
        
        # Code
        'py': '🐍',
        'ipynb': '📓',
        'r': '📊',
        'm': '📊',
        
        # Archives
        'zip': '🗜️',
        'tar': '🗜️',
        'gz': '🗜️',
        'rar': '🗜️',
        
        # Documents
        'pdf': '📑',
        'doc': '📝',
        'docx': '📝',
    }
    
    return icon_map.get(ext, '📄')


def build_tree_markdown(node: Dict, depth: int = 0, max_depth: int = 5) -> str:
    """
    Build a markdown string representation of the directory tree.
    
    Args:
        node: Tree node dictionary
        depth: Current depth for indentation
        max_depth: Maximum depth to traverse
        
    Returns:
        str: Markdown formatted tree
    """
    if depth >= max_depth:
        return ""
    
    markdown = ""
    indent = "  " * depth
    
    if node["type"] == "directory" and "children" in node:
        # Sort children: directories first, then files
        dirs = [c for c in node["children"] if c["type"] == "directory"]
        files = [c for c in node["children"] if c["type"] == "file"]
        
        # Add directories
        for child in dirs:
            folder_icon = "📁" if child.get("children") else "📂"
            markdown += f"{indent}- {folder_icon} **{child['name']}**\n"
            markdown += build_tree_markdown(child, depth + 1, max_depth)
        
        # Add files
        for file in files:
            file_icon = get_file_icon(file["name"])
            size_str = format_size(file["size"])
            markdown += f"{indent}- {file_icon} `{file['name']}` ({size_str})\n"
    
    return markdown


def render_tree_node(node: Dict, key_prefix: str = "", depth: int = 0) -> None:
    """
    Render a tree node with Streamlit components.
    Only uses expanders for top-level directories (depth 0).
    
    Args:
        node: Tree node dictionary
        key_prefix: Unique key prefix for Streamlit components
        depth: Current depth for indentation
    """
    if node["type"] == "directory":
        # Use emoji for folders
        folder_icon = "📁" if node.get("children") else "📂"
        
        # Only use expanders for top-level directories
        if depth == 0:
            # Create expander for top-level directories - COLLAPSED by default
            with st.expander(f"{folder_icon} **{node['name']}**", expanded=False):
                if "error" in node:
                    st.error(f"Error: {node['error']}")
                elif node.get("children"):
                    # Build markdown for the entire subtree
                    tree_markdown = build_tree_markdown(node)
                    if tree_markdown:
                        st.markdown(tree_markdown)
                    else:
                        st.caption("Empty directory")
                else:
                    st.caption("Empty directory")
        else:
            # For non-top-level directories, this shouldn't be called
            # but if it is, just render as markdown
            st.markdown(f"{'  ' * depth}- {folder_icon} **{node['name']}**")
    
    elif node["type"] == "error":
        st.error(f"❌ {node['name']}")


def collect_files(node: Dict, files_list: List[Dict]) -> None:
    """Recursively collect all files from a directory tree."""
    if node["type"] == "file":
        files_list.append(node)
    elif node["type"] == "directory" and "children" in node:
        for child in node["children"]:
            collect_files(child, files_list)


def get_current_sandbox_path(session_state: Dict) -> Optional[str]:
    """
    Extract the current sandbox directory path from session state.
    
    Args:
        session_state: Streamlit session state
        
    Returns:
        str or None: Path to current sandbox directory
    """
    if "active_datasets" in session_state and session_state["active_datasets"]:
        # Get the first active dataset's path
        first_doi = next(iter(session_state["active_datasets"]))
        cached_data = session_state["datasets_cache"].get(first_doi)
        if cached_data and cached_data[0]:
            dataset_path = cached_data[0]
            if isinstance(dataset_path, str) and os.path.isdir(dataset_path):
                # Get the parent sandbox directory
                return os.path.dirname(os.path.abspath(dataset_path))
    return None


def render_directory_explorer() -> None:
    """
    Main function to render the complete directory explorer UI.
    This should be called from app.py in the data agent page section.
    """
    # --- Directory Explorer Section ---
    st.markdown("---")  # Separator

    # Directory Explorer header
    st.subheader("📁 Directory Explorer")
    st.caption("View the contents of the plotting data folder and current sandbox directory")

    # Get current sandbox directory
    current_sandbox = get_current_sandbox_path(st.session_state)

    # Create the main directory explorer container
    with st.container():
        # Create tabs for different directories
        tab1, tab2, tab3 = st.tabs(["🌍 Plotting Data", "📦 Current Sandbox", "📊 Results"])
        
        # Tab 1: Plotting Data Directory
        with tab1:
            plotting_data_path = os.path.join("data", "plotting_data")
            
            if os.path.exists(plotting_data_path):
                # Add refresh button
                if st.button("🔄 Refresh", key="refresh_plotting_data"):
                    st.rerun()
                
                # Get and display directory tree
                tree = get_directory_tree(plotting_data_path, max_depth=5)
                
                # Display some statistics
                st.caption(f"📊 Total items: {len(tree.get('children', []))}")
                
                # Render with expanders
                render_tree_node(tree, "plotting_data")
            else:
                st.warning("⚠️ Plotting data directory not found")
                st.caption(f"Expected path: {plotting_data_path}")
        
        # Tab 2: Current Sandbox Directory
        with tab2:
            if current_sandbox and os.path.exists(current_sandbox):
                # Add refresh button and path display
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🔄 Refresh", key="refresh_sandbox"):
                        st.rerun()
                with col2:
                    st.caption(f"📍 Path: `{current_sandbox}`")
                
                # Get and display directory tree
                sandbox_tree = get_directory_tree(current_sandbox, max_depth=5)
                
                # Display sandbox info
                if "era5_data" in current_sandbox:
                    st.info("🌍 This is an ERA5 data sandbox")
                elif "copernicus_data" in current_sandbox:
                    st.info("🌊 This is a Copernicus Marine data sandbox")
                
                # Show total size
                total_size = sum(child.get("size", 0) for child in sandbox_tree.get("children", []) if child["type"] == "file")
                st.caption(f"📊 Total size: {format_size(total_size)} | Items: {len(sandbox_tree.get('children', []))}")
                
                # Render with expanders
                render_tree_node(sandbox_tree, "sandbox")
                
                # Add download functionality for individual files
                st.markdown("---")
                st.markdown("**Quick Actions:**")
                
                # List all files for quick download
                all_files = []
                collect_files(sandbox_tree, all_files)
                
                if all_files:
                    selected_file = st.selectbox(
                        "Select a file to download:",
                        options=[f["name"] for f in all_files],
                        key="file_download_select"
                    )
                    
                    if selected_file:
                        file_info = next(f for f in all_files if f["name"] == selected_file)
                        if st.button(f"⬇️ Download {selected_file}", key="download_selected_file"):
                            try:
                                with open(file_info["path"], "rb") as f:
                                    file_data = f.read()
                                st.download_button(
                                    label=f"💾 Save {selected_file}",
                                    data=file_data,
                                    file_name=selected_file,
                                    mime="application/octet-stream",
                                    key=f"download_{selected_file}"
                                )
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")
            else:
                st.info("📦 No active sandbox directory")
                st.caption("Load a dataset to see its sandbox directory contents")
        
        # Tab 3: Results Directory
        with tab3:
            if current_sandbox and os.path.exists(current_sandbox):
                results_path = os.path.join(current_sandbox, "results")
                
                # Add refresh button
                if st.button("🔄 Refresh", key="refresh_results"):
                    st.rerun()
                
                if os.path.exists(results_path):
                    # Get and display directory tree for results
                    results_tree = get_directory_tree(results_path, max_depth=3)
                    
                    # Count visualizations
                    all_result_files = []
                    collect_files(results_tree, all_result_files)
                    
                    # Filter for image files
                    image_files = [f for f in all_result_files if f["name"].endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf'))]
                    
                    st.caption(f"📊 Total visualizations: {len(image_files)}")
                    
                    # Display the tree
                    if results_tree.get("children"):
                        st.markdown("### 📊 Generated Visualizations")
                        
                        # Show images in a grid
                        if image_files:
                            # Create columns for image gallery
                            cols = st.columns(3)
                            for idx, img_file in enumerate(image_files):
                                with cols[idx % 3]:
                                    if img_file["name"].endswith(('.png', '.jpg', '.jpeg')):
                                        try:
                                            st.image(img_file["path"], caption=img_file["name"], use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Error loading image: {str(e)}")
                                    else:
                                        st.info(f"📄 {img_file['name']}")
                                    
                                    # Add download button for each file
                                    with open(img_file["path"], "rb") as f:
                                        file_data = f.read()
                                    st.download_button(
                                        label=f"⬇️ Download",
                                        data=file_data,
                                        file_name=img_file["name"],
                                        mime="application/octet-stream",
                                        key=f"download_result_{img_file['name']}"
                                    )
                        
                        # Also show the tree structure
                        with st.expander("📁 File Structure", expanded=False):
                            tree_markdown = build_tree_markdown(results_tree)
                            if tree_markdown:
                                st.markdown(tree_markdown)
                    else:
                        st.info("📊 No results generated yet")
                        st.caption("Visualizations created by agents will appear here")
                else:
                    st.info("📊 No results directory yet")
                    st.caption("The results folder will be created when visualizations are generated")
            else:
                st.info("📦 No active sandbox directory")
                st.caption("Load a dataset to see visualization results")

    # Add auto-refresh functionality
    if st.session_state.get("processing", False):
        # Auto-refresh while processing
        st.caption("🔄 Directory view will update automatically when processing completes...")
        
    # Add a visual separator before the next section
    st.markdown("---")
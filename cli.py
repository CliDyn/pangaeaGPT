#!/usr/bin/env python3
"""
cli.py - Command Line Interface for PANGAEA GPT

This provides a command-line interface to the PANGAEA GPT system,
allowing users to search datasets and run analysis without the Streamlit UI.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CRITICAL: Set deployment mode BEFORE any imports that use streamlit
# This prevents src/config.py from trying to access st.secrets
os.environ["PANGAEA_CLI_MODE"] = "true"
os.environ["DEPLOYMENT_MODE"] = "cli"  # Override deployment mode for CLI

# Import CLI utilities and set up mock Streamlit BEFORE any other imports
from src.cli_utils import setup_cli_mode, CLICallbackHandler

# Set up CLI mode - this must happen before importing any modules that use streamlit
mock_st = setup_cli_mode()

# Now we can import the rest of the modules
from src.agents import create_search_agent, create_supervisor_agent
from src.search.search_pg_default import pg_search_default, direct_access_doi
from src.memory import CustomMemorySaver
from main import (
    get_datasets_info_for_active_datasets,
    load_selected_datasets_into_cache,
    set_active_datasets_from_selection
)


def make_json_serializable(obj):
    """
    Recursively convert any object to a JSON-serializable format.
    Handles nested structures, pandas objects, numpy arrays, etc.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, date
    
    # Handle None
    if obj is None:
        return None
    
    # Handle basic JSON-serializable types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle datetime objects
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    # Handle pandas Series
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    
    # Handle pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    
    # Handle numpy arrays and numpy scalars
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    
    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return list(obj)
    
    # For any other object, try to convert to dict or string
    try:
        # Try to get __dict__ attribute
        if hasattr(obj, '__dict__'):
            return make_json_serializable(obj.__dict__)
    except:
        pass
    
    # Last resort: convert to string
    return str(obj)


class CLIInterface:
    """Command Line Interface for PANGAEA GPT"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1", 
                 anthropic_key: Optional[str] = None,
                 langchain_key: Optional[str] = None,
                 langchain_project: Optional[str] = None):
        """Initialize CLI interface with API keys"""
        # Set up environment variables
        os.environ["OPENAI_API_KEY"] = api_key
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        if langchain_key:
            os.environ["LANGCHAIN_API_KEY"] = langchain_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if langchain_project:
                os.environ["LANGCHAIN_PROJECT_NAME"] = langchain_project
        
        # Initialize session state properly using the mock from cli_utils
        import streamlit as st
        
        # Initialize all required session state keys
        default_state = {
            "messages_search": [],
            "messages_data_agent": [],
            "datasets_cache": {},
            "datasets_info": None,
            "active_datasets": [],
            "selected_datasets": set(),
            "show_dataset": True,
            "current_page": "search",
            "dataset_dfs": {},
            "dataset_names": {},
            "saved_plot_paths": {},
            "memory": CustomMemorySaver(),
            "oceanographer_agent_used": False,
            "ecologist_agent_used": False,
            "visualization_agent_used": False,
            "dataframe_agent_used": False,
            "specialized_agent_used": False,
            "search_method": "PANGAEA Search (default)",
            "selected_text": "",
            "new_plot_generated": False,
            "execution_history": [],
            "model_name": model_name,
            "search_mode": "simple",
            "processing": False,
            "intermediate_steps": [],
            "thread_id": None,
            "viz_datasets_text": "",
            "search_results_cache": {}
        }
        
        # Update the mock session state
        for key, value in default_state.items():
            st.session_state[key] = value
        
        # Keep a reference to session state
        self.session_state = st.session_state
        
        # Also mock secrets if API keys are provided
        if api_key:
            mock_st.secrets.data["general"]["openai_api_key"] = api_key
        if anthropic_key:
            mock_st.secrets.data["general"]["anthropic_api_key"] = anthropic_key
        if langchain_key:
            mock_st.secrets.data["general"]["langchain_api_key"] = langchain_key
        if langchain_project:
            mock_st.secrets.data["general"]["langchain_project_name"] = langchain_project
        
        # Set up logging
        self.setup_logging()
        
        # Create output directory
        self.output_dir = Path("cli_output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"CLI Interface initialized. Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('pangaea_cli.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def search_datasets(self, query: str, count: int = 15, 
                       search_mode: str = "simple",
                       mindate: Optional[str] = None,
                       maxdate: Optional[str] = None,
                       minlat: Optional[float] = None,
                       maxlat: Optional[float] = None,
                       minlon: Optional[float] = None,
                       maxlon: Optional[float] = None) -> pd.DataFrame:
        """Search for datasets using the search agent"""
        print(f"\n🔍 Searching for: '{query}' (mode: {search_mode})")
        
        self.session_state["search_mode"] = search_mode
        
        # Create search agent
        search_agent = create_search_agent(
            datasets_info=self.session_state["datasets_info"],
            search_mode=search_mode
        )
        
        # Process the search query
        from main import process_search_query
        result = process_search_query(query, search_agent, self.session_state)
        
        print(f"\n✅ {result}")
        
        # Get the datasets info
        datasets_info = self.session_state.get("datasets_info")
        
        if datasets_info is not None and not datasets_info.empty:
            # Save results to CSV
            output_file = self.output_dir / "search_results.csv"
            datasets_info.to_csv(output_file, index=False)
            print(f"\n📄 Search results saved to: {output_file}")
            
            # Display results
            print("\n📊 Search Results:")
            for idx, row in datasets_info.iterrows():
                print(f"\n{idx + 1}. {row['Name']}")
                print(f"   DOI: {row['DOI']}")
                print(f"   Description: {row['Short Description'][:200]}...")
                print(f"   Parameters: {row['Parameters'][:100]}...")
            
            return datasets_info
        else:
            print("\n⚠️ No datasets found.")
            return pd.DataFrame()
    
    def direct_access_datasets(self, dois: List[str]) -> pd.DataFrame:
        """Direct access to datasets by DOI"""
        print(f"\n📥 Directly accessing DOIs: {', '.join(dois)}")
        
        # Join DOIs with commas for the function
        doi_string = ', '.join(dois)
        datasets_info, prompt_text = direct_access_doi(doi_string)
        
        if not datasets_info.empty:
            # Save results
            output_file = self.output_dir / "direct_access_results.csv"
            datasets_info.to_csv(output_file, index=False)
            print(f"\n📄 Results saved to: {output_file}")
            
            # Display results
            print("\n📊 Direct Access Results:")
            for idx, row in datasets_info.iterrows():
                print(f"\n{idx + 1}. {row['Name']}")
                print(f"   DOI: {row['DOI']}")
                print(f"   Parameters: {row['Parameters']}")
            
            return datasets_info
        else:
            print("\n⚠️ Failed to access datasets.")
            return pd.DataFrame()
    
    def analyze_datasets(self, dataset_indices: List[int], query: str,
                        datasets_info: Optional[pd.DataFrame] = None):
        """Analyze selected datasets with the data agent"""
        if datasets_info is None:
            datasets_info = self.session_state.get("datasets_info")
        
        if datasets_info is None or datasets_info.empty:
            print("\n⚠️ No datasets available. Please search first.")
            return
        
        # CRITICAL FIX: Clear all agent-related session state before new analysis
        print("\n🔄 Clearing previous analysis state...")
        
        # Clear agent usage flags
        self.session_state["oceanographer_agent_used"] = False
        self.session_state["ecologist_agent_used"] = False
        self.session_state["visualization_agent_used"] = False
        self.session_state["dataframe_agent_used"] = False
        self.session_state["specialized_agent_used"] = False
        
        # Clear messages and conversation history
        self.session_state["messages_data_agent"] = []
        self.session_state["intermediate_steps"] = []
        
        # Generate new thread ID for this analysis
        import uuid
        self.session_state["thread_id"] = str(uuid.uuid4())
        
        # Clear any cached plot info
        self.session_state["new_plot_generated"] = False
        self.session_state["saved_plot_paths"] = {}
        
        # Reset processing flags
        self.session_state["processing"] = False
        
        print(f"✓ Session state cleared for new analysis: '{query}'")
        
        # Select datasets
        selected_dois = set()
        for idx in dataset_indices:
            if 0 <= idx - 1 < len(datasets_info):
                doi = datasets_info.iloc[idx - 1]['DOI']
                selected_dois.add(doi)
                print(f"✓ Selected dataset {idx}: {datasets_info.iloc[idx - 1]['Name']}")
        
        if not selected_dois:
            print("\n⚠️ No valid datasets selected.")
            return
        
        self.session_state["selected_datasets"] = selected_dois
        
        # Load datasets
        print("\n📥 Loading selected datasets...")
        load_selected_datasets_into_cache(selected_dois, self.session_state)
        set_active_datasets_from_selection(self.session_state)

        # Get dataset info for active datasets
        active_datasets_info = get_datasets_info_for_active_datasets(self.session_state)

        if not active_datasets_info:
            print("\n⚠️ Failed to load datasets.")
            return

        # Create and invoke supervisor agent
        print(f"\n🤖 Analyzing with query: '{query}'")

        # FIX: Add the user's query to the message history before invoking the agent
        self.session_state["messages_data_agent"].append({"role": "user", "content": query})

        logging.info(f"CLI Analysis Query: {query}")
        logging.info(f"Selected DOIs: {', '.join(selected_dois)}")
        logging.info(f"Active datasets count: {len(active_datasets_info)}")
        logging.info(f"Thread ID: {self.session_state.get('thread_id', 'None')}")
        
        # Verify we have clean session state
        agent_flags = {
            "oceanographer_agent_used": self.session_state.get("oceanographer_agent_used", False),
            "ecologist_agent_used": self.session_state.get("ecologist_agent_used", False), 
            "visualization_agent_used": self.session_state.get("visualization_agent_used", False),
            "dataframe_agent_used": self.session_state.get("dataframe_agent_used", False)
        }
        logging.info(f"Agent flags before analysis: {agent_flags}")
        
        # Use the standard create_and_invoke_supervisor_agent - no monkey-patching
        from main import create_and_invoke_supervisor_agent
        
        print(f"✓ Using supervisor agent with planning tool")
        
        # Call the standard function without any modifications
        response = create_and_invoke_supervisor_agent(
            user_query=query,
            datasets_info=active_datasets_info,
            memory=self.session_state["memory"],
            session_data=self.session_state,
            st_callback=None
        )
        
        # Rest of the method remains the same...
        if response:
            # Extract the final message
            messages = response.get('messages', [])
            if messages:
                last_message = messages[-1]
                logging.info(f"Final message from agent: {last_message.name if hasattr(last_message, 'name') else 'Unknown'}")
                logging.info(f"Message content preview: {last_message.content[:200]}...")
                
                print(f"\n📊 Analysis Result:\n")
                print(last_message.content)
                
                # Save result
                output_file = self.output_dir / "analysis_result.txt"
                with open(output_file, 'w') as f:
                    f.write(f"Query: {query}\n")
                    f.write(f"Selected DOIs: {', '.join(selected_dois)}\n\n")
                    f.write("Analysis Result:\n")
                    f.write(last_message.content)
                
                print(f"\n📄 Analysis saved to: {output_file}")
                
                # Check for generated plots
                plot_images = response.get('plot_images', [])
                if plot_images:
                    print(f"\n🎨 Generated {len(plot_images)} plots:")
                    for plot_path in plot_images:
                        if os.path.exists(plot_path):
                            print(f"   - {plot_path}")
                
                # Save execution history
                history_file = self.output_dir / "execution_history.json"
                try:
                    serializable_history = make_json_serializable(self.session_state.get("execution_history", []))
                    with open(history_file, 'w') as f:
                        json.dump(serializable_history, f, indent=2, default=str)
                    print(f"\n📄 Execution history saved to: {history_file}")
                except Exception as e:
                    logging.error(f"Could not save execution history: {e}")
        else:
            print("\n⚠️ Analysis failed.")
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n🌊 PANGAEA GPT - Interactive CLI Mode")
        print("=" * 50)
        print("Commands:")
        print("  search <query>     - Search for datasets")
        print("  direct <doi1,doi2> - Direct access to DOIs")
        print("  select <1,2,3>     - Select datasets by number")
        print("  analyze <query>    - Analyze selected datasets")
        print("  list               - List current datasets")
        print("  clear              - Clear current session")
        print("  help               - Show this help")
        print("  exit               - Exit the program")
        print("=" * 50)
        
        # Store the original model name for session resets
        original_model = self.session_state.get("model_name", "gpt-4.1")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if not command:
                    continue
                
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd == "exit":
                    print("\n👋 Goodbye!")
                    break
                
                elif cmd == "help":
                    self.interactive_mode()  # Show help again
                
                elif cmd == "search":
                    if not args:
                        print("⚠️ Please provide a search query.")
                        continue
                    self.search_datasets(args)
                
                elif cmd == "direct":
                    if not args:
                        print("⚠️ Please provide DOIs separated by commas.")
                        continue
                    dois = [doi.strip() for doi in args.split(',')]
                    self.direct_access_datasets(dois)
                
                elif cmd == "select":
                    if not args:
                        print("⚠️ Please provide dataset numbers separated by commas.")
                        continue
                    try:
                        indices = [int(x.strip()) for x in args.split(',')]
                        # Mark as selected
                        datasets_info = self.session_state.get("datasets_info")
                        if datasets_info is not None:
                            for idx in indices:
                                if 0 <= idx - 1 < len(datasets_info):
                                    doi = datasets_info.iloc[idx - 1]['DOI']
                                    self.session_state["selected_datasets"].add(doi)
                            print(f"✓ Selected {len(indices)} datasets")
                        else:
                            print("⚠️ No datasets available to select from.")
                    except ValueError:
                        print("⚠️ Invalid dataset numbers.")
                
                elif cmd == "analyze":
                    if not args:
                        print("⚠️ Please provide an analysis query.")
                        continue
                    selected = list(self.session_state.get("selected_datasets", set()))
                    if not selected:
                        print("⚠️ No datasets selected. Use 'select' command first.")
                        continue
                    # Get indices of selected datasets
                    datasets_info = self.session_state.get("datasets_info")
                    if datasets_info is not None:
                        indices = []
                        for idx, row in datasets_info.iterrows():
                            if row['DOI'] in selected:
                                indices.append(idx + 1)
                        self.analyze_datasets(indices, args, datasets_info)
                
                elif cmd == "list":
                    datasets_info = self.session_state.get("datasets_info")
                    selected = self.session_state.get("selected_datasets", set())
                    if datasets_info is not None and not datasets_info.empty:
                        print("\n📊 Current Datasets:")
                        for idx, row in datasets_info.iterrows():
                            marker = "✓" if row['DOI'] in selected else " "
                            print(f"\n[{marker}] {idx + 1}. {row['Name']}")
                            print(f"     DOI: {row['DOI']}")
                    else:
                        print("\n⚠️ No datasets loaded.")
                
                elif cmd == "clear":
                    # Clear session state while preserving model
                    import streamlit as st
                    import uuid
                    preserved_model = st.session_state.get("model_name", original_model)
                    
                    # Reinitialize with default values
                    default_state = {
                        "messages_search": [],
                        "messages_data_agent": [],
                        "datasets_cache": {},
                        "datasets_info": None,
                        "active_datasets": [],
                        "selected_datasets": set(),
                        "show_dataset": True,
                        "current_page": "search",
                        "dataset_dfs": {},
                        "dataset_names": {},
                        "saved_plot_paths": {},
                        "memory": CustomMemorySaver(),
                        "oceanographer_agent_used": False,
                        "ecologist_agent_used": False,
                        "visualization_agent_used": False,
                        "dataframe_agent_used": False,
                        "specialized_agent_used": False,
                        "search_method": "PANGAEA Search (default)",
                        "selected_text": "",
                        "new_plot_generated": False,
                        "execution_history": [],
                        "model_name": preserved_model,
                        "search_mode": "simple",
                        "processing": False,
                        "intermediate_steps": [],
                        "thread_id": str(uuid.uuid4()),  # Generate new thread ID
                        "viz_datasets_text": "",
                        "search_results_cache": {}
                    }
                    
                    # Clear and update
                    st.session_state.clear()
                    st.session_state.update(default_state)
                    self.session_state = st.session_state
                    
                    print("✓ Session cleared with fresh thread ID.")
                
                else:
                    print(f"⚠️ Unknown command: '{cmd}'. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                logging.error(f"Error in interactive mode: {str(e)}", exc_info=True)


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="PANGAEA GPT - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for datasets
  python cli.py --search "temperature arctic ocean" --count 20
  
  # Direct access to DOIs
  python cli.py --direct-doi "10.1594/PANGAEA.123456,10.1594/PANGAEA.789012"
  
  # Search and analyze in one command
  python cli.py --search "salinity fram strait" --analyze "Plot salinity distribution" --select 1,2,3
  
  # Interactive mode
  python cli.py --interactive
  
  # Use different model
  python cli.py --model gpt-4o-mini --search "biodiversity data"
        """
    )
    
    # API keys
    parser.add_argument('--api-key', type=str, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--anthropic-key', type=str, help='Anthropic API key (optional)')
    parser.add_argument('--langchain-key', type=str, help='LangChain API key (optional)')
    parser.add_argument('--langchain-project', type=str, help='LangChain project name (optional)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='gpt-4.1-mini',
                       choices=['gpt-5', 'gpt-4.1-mini', 'gpt-4.1', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o3-mini'],
                       help='Model to use (default: gpt-4.1)')
    
    # Search options
    parser.add_argument('--search', type=str, help='Search query for datasets')
    parser.add_argument('--search-mode', type=str, default='simple', choices=['simple', 'deep'],
                       help='Search mode: simple (fast) or deep (comprehensive)')
    parser.add_argument('--count', type=int, default=15, help='Number of search results (default: 15)')
    parser.add_argument('--mindate', type=str, help='Minimum date (YYYY-MM-DD)')
    parser.add_argument('--maxdate', type=str, help='Maximum date (YYYY-MM-DD)')
    parser.add_argument('--minlat', type=float, help='Minimum latitude')
    parser.add_argument('--maxlat', type=float, help='Maximum latitude')
    parser.add_argument('--minlon', type=float, help='Minimum longitude')
    parser.add_argument('--maxlon', type=float, help='Maximum longitude')
    
    # Direct DOI access
    parser.add_argument('--direct-doi', type=str, help='Direct access to DOIs (comma-separated)')
    
    # Analysis options
    parser.add_argument('--select', type=str, help='Select datasets by number (comma-separated, e.g., 1,2,3)')
    parser.add_argument('--analyze', type=str, help='Analysis query for selected datasets')
    
    # Interactive mode
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    # Output options
    parser.add_argument('--output-dir', type=str, help='Output directory (default: cli_output/timestamp)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OpenAI API key required. Set via --api-key or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Get other API keys
    anthropic_key = args.anthropic_key or os.environ.get('ANTHROPIC_API_KEY')
    langchain_key = args.langchain_key or os.environ.get('LANGCHAIN_API_KEY')
    langchain_project = args.langchain_project or os.environ.get('LANGCHAIN_PROJECT_NAME')
    
    # Create CLI interface
    cli = CLIInterface(
        api_key=api_key,
        model_name=args.model,
        anthropic_key=anthropic_key,
        langchain_key=langchain_key,
        langchain_project=langchain_project
    )
    
    # Override output directory if specified
    if args.output_dir:
        cli.output_dir = Path(args.output_dir)
        cli.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run in interactive mode
    if args.interactive:
        cli.interactive_mode()
        return
    
    # Execute commands
    datasets_info = None
    
    # Search or direct DOI access
    if args.search:
        datasets_info = cli.search_datasets(
            query=args.search,
            count=args.count,
            search_mode=args.search_mode,
            mindate=args.mindate,
            maxdate=args.maxdate,
            minlat=args.minlat,
            maxlat=args.maxlat,
            minlon=args.minlon,
            maxlon=args.maxlon
        )
    elif args.direct_doi:
        dois = [doi.strip() for doi in args.direct_doi.split(',')]
        datasets_info = cli.direct_access_datasets(dois)
    
    # Select and analyze
    if args.select and args.analyze:
        if datasets_info is None or datasets_info.empty:
            print("\n⚠️ No datasets available. Please use --search or --direct-doi first.")
            sys.exit(1)
        
        try:
            indices = [int(x.strip()) for x in args.select.split(',')]
            cli.analyze_datasets(indices, args.analyze, datasets_info)
        except ValueError:
            print("❌ Error: Invalid dataset selection. Use comma-separated numbers (e.g., 1,2,3)")
            sys.exit(1)
    elif args.analyze:
        print("\n⚠️ --analyze requires --select to specify which datasets to analyze.")
        sys.exit(1)
    
    # If no analysis requested but datasets were found, just show summary
    if datasets_info is not None and not args.analyze:
        print(f"\n✅ Found {len(datasets_info)} datasets. Use --select and --analyze to analyze them.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Command-line interface for data import pipeline.

This script provides a simple CLI for importing academic papers and citations
from Semantic Scholar using the data import pipeline.

Usage Examples:
    # Import papers by search query
    python -m src.cli.import_data search "machine learning" --max-papers 100

    # Import specific paper IDs
    python -m src.cli.import_data ids paper1.txt --batch-size 20
    
    # Import with filtering options
    python -m src.cli.import_data search "neural networks" --min-citations 10 --year-range 2020 2024
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List
import signal
import time

from ..data.import_pipeline import (
    ImportConfiguration,
    DataImportPipeline,
    ImportStatus,
    quick_import_by_search,
    quick_import_by_ids
)
from ..utils.validation import validate_import_configuration


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('import.log')
        ]
    )


def load_paper_ids_from_file(file_path: str) -> List[str]:
    """Load paper IDs from a text file (one per line)."""
    try:
        with open(file_path, 'r') as f:
            paper_ids = [line.strip() for line in f if line.strip()]
        return paper_ids
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)


def print_progress_update(progress):
    """Print progress updates to console."""
    status_symbols = {
        ImportStatus.PENDING: "⏳",
        ImportStatus.IN_PROGRESS: "🔄",
        ImportStatus.COMPLETED: "✅",
        ImportStatus.FAILED: "❌",
        ImportStatus.CANCELLED: "🛑",
        ImportStatus.PAUSED: "⏸️"
    }
    
    symbol = status_symbols.get(progress.status, "❓")
    elapsed = str(progress.elapsed_time).split('.')[0] if progress.elapsed_time else "00:00:00"
    
    if progress.total_papers > 0:
        papers_percent = progress.papers_progress_percent
        print(f"\r{symbol} Papers: {progress.processed_papers}/{progress.total_papers} ({papers_percent:.1f}%) | "
              f"Created: {progress.papers_created} papers, {progress.citations_created} citations | "
              f"Elapsed: {elapsed}", end="", flush=True)
    else:
        print(f"\r{symbol} Status: {progress.status.value} | Elapsed: {elapsed}", end="", flush=True)


def import_by_search(args):
    """Handle search-based import."""
    print(f"🔍 Importing papers with search query: '{args.query}'")
    print(f"📊 Configuration: max_papers={args.max_papers}, batch_size={args.batch_size}")
    
    # Create configuration
    config = ImportConfiguration(
        search_query=args.query,
        max_papers=args.max_papers,
        batch_size=args.batch_size,
        include_citations=args.include_citations,
        include_authors=args.include_authors,
        include_venues=args.include_venues,
        min_citation_count=args.min_citations,
        year_range=tuple(args.year_range) if args.year_range else None,
        api_delay=args.api_delay,
        save_progress=True
    )
    
    # Validate configuration
    validation = validate_import_configuration(config)
    if not validation['is_valid']:
        print("❌ Configuration is invalid:")
        for error in validation['errors']:
            print(f"  • {error}")
        sys.exit(1)
    
    if validation['warnings']:
        print("⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"  • {warning}")
        print()
    
    # Run import with progress tracking
    pipeline = DataImportPipeline(config)
    pipeline.add_progress_callback(print_progress_update)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Import cancelled by user")
        pipeline.cancel_import()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        progress = pipeline.import_papers_by_search(args.query, args.max_papers)
        
        print()  # New line after progress updates
        
        if progress.status == ImportStatus.COMPLETED:
            print("✅ Import completed successfully!")
        elif progress.status == ImportStatus.CANCELLED:
            print("🛑 Import was cancelled")
        else:
            print(f"❌ Import failed: {progress.status.value}")
        
        # Print summary
        print(f"\n📊 Import Summary:")
        print(f"  Papers processed: {progress.processed_papers}")
        print(f"  Papers created: {progress.papers_created}")
        print(f"  Citations created: {progress.citations_created}")
        print(f"  Authors created: {progress.authors_created}")
        print(f"  Venues created: {progress.venues_created}")
        
        if progress.elapsed_time:
            print(f"  Total time: {str(progress.elapsed_time).split('.')[0]}")
        
        if progress.errors:
            print(f"  Errors: {len(progress.errors)}")
            if args.verbose:
                for error in progress.errors:
                    print(f"    • {error}")
        
        if progress.warnings:
            print(f"  Warnings: {len(progress.warnings)}")
            if args.verbose:
                for warning in progress.warnings:
                    print(f"    • {warning}")
    
    except Exception as e:
        print(f"\n❌ Import failed with exception: {e}")
        sys.exit(1)


def import_by_ids(args):
    """Handle ID-based import."""
    # Load paper IDs
    if args.ids_file:
        paper_ids = load_paper_ids_from_file(args.ids_file)
        print(f"📋 Loaded {len(paper_ids)} paper IDs from {args.ids_file}")
    else:
        paper_ids = args.ids
        print(f"📋 Importing {len(paper_ids)} papers by ID")
    
    if not paper_ids:
        print("❌ No paper IDs provided")
        sys.exit(1)
    
    # Create configuration
    config = ImportConfiguration(
        paper_ids=paper_ids,
        max_papers=len(paper_ids),
        batch_size=args.batch_size,
        include_citations=args.include_citations,
        include_authors=args.include_authors,
        include_venues=args.include_venues,
        api_delay=args.api_delay,
        save_progress=True
    )
    
    # Run import
    pipeline = DataImportPipeline(config)
    pipeline.add_progress_callback(print_progress_update)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Import cancelled by user")
        pipeline.cancel_import()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        progress = pipeline.import_papers_by_ids(paper_ids)
        
        print()  # New line after progress updates
        
        if progress.status == ImportStatus.COMPLETED:
            print("✅ Import completed successfully!")
        elif progress.status == ImportStatus.CANCELLED:
            print("🛑 Import was cancelled")
        else:
            print(f"❌ Import failed: {progress.status.value}")
        
        # Print summary
        print(f"\n📊 Import Summary:")
        print(f"  Papers processed: {progress.processed_papers}")
        print(f"  Papers created: {progress.papers_created}")
        print(f"  Citations created: {progress.citations_created}")
        print(f"  Authors created: {progress.authors_created}")
        print(f"  Venues created: {progress.venues_created}")
        
        if progress.elapsed_time:
            print(f"  Total time: {str(progress.elapsed_time).split('.')[0]}")
    
    except Exception as e:
        print(f"\n❌ Import failed with exception: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Import academic papers and citations from Semantic Scholar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search "machine learning" --max-papers 100
  %(prog)s ids paper1 paper2 paper3 --batch-size 10
  %(prog)s ids --ids-file paper_ids.txt --include-citations
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of papers to process in each batch (default: 50)'
    )
    
    parser.add_argument(
        '--api-delay',
        type=float,
        default=1.0,
        help='Delay between API requests in seconds (default: 1.0)'
    )
    
    # Include/exclude options
    parser.add_argument(
        '--no-citations',
        dest='include_citations',
        action='store_false',
        default=True,
        help='Skip citation relationships'
    )
    
    parser.add_argument(
        '--no-authors',
        dest='include_authors',
        action='store_false',
        default=True,
        help='Skip author information'
    )
    
    parser.add_argument(
        '--no-venues',
        dest='include_venues',
        action='store_false',
        default=True,
        help='Skip venue information'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Import method')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Import papers by search query')
    search_parser.add_argument('query', help='Search query string')
    search_parser.add_argument(
        '--max-papers',
        type=int,
        default=100,
        help='Maximum number of papers to import (default: 100)'
    )
    search_parser.add_argument(
        '--min-citations',
        type=int,
        default=0,
        help='Minimum citation count filter (default: 0)'
    )
    search_parser.add_argument(
        '--year-range',
        type=int,
        nargs=2,
        metavar=('START_YEAR', 'END_YEAR'),
        help='Publication year range (e.g., --year-range 2020 2024)'
    )
    
    # IDs command  
    ids_parser = subparsers.add_parser('ids', help='Import papers by specific IDs')
    
    # Either specify IDs directly or load from file
    ids_group = ids_parser.add_mutually_exclusive_group(required=True)
    ids_group.add_argument(
        '--ids',
        nargs='+',
        help='Paper IDs to import'
    )
    ids_group.add_argument(
        '--ids-file',
        help='File containing paper IDs (one per line)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Route to appropriate function
    if args.command == 'search':
        import_by_search(args)
    elif args.command == 'ids':
        import_by_ids(args)


if __name__ == '__main__':
    main()
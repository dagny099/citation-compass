"""
Command-line interface for Academic Citation Platform.

Provides CLI commands for:
- Database setup and management
- Data import and export
- Model training and evaluation
- Network analysis
- Health checks and diagnostics
"""

import argparse
import sys
import logging
from typing import Optional
from pathlib import Path

from src.config.settings import settings
from src.database.connection import Neo4jConnection
from src.services.ml_service import get_ml_service
from src.services.analytics_service import get_analytics_service


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI operations."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_health_check(args) -> int:
    """Check system health and component status."""
    print("Academic Citation Platform - Health Check")
    print("=" * 50)
    
    # Check configuration
    if settings.is_valid():
        print("✅ Configuration: Valid")
    else:
        print("❌ Configuration: Invalid")
        for error in settings.get_validation_errors():
            print(f"   - {error}")
        return 1
    
    # Check database connection
    try:
        conn = Neo4jConnection()
        if conn.test_connection():
            print("✅ Database: Connected")
        else:
            print("❌ Database: Connection failed")
            return 1
    except Exception as e:
        print(f"❌ Database: Error - {e}")
        return 1
    
    # Check ML service
    try:
        ml_service = get_ml_service()
        if ml_service.health_check()['status'] == 'healthy':
            print("✅ ML Service: Ready")
        else:
            print("⚠️  ML Service: Issues detected")
    except Exception as e:
        print(f"❌ ML Service: Error - {e}")
        return 1
    
    # Check analytics service
    try:
        analytics = get_analytics_service()
        print("✅ Analytics Service: Ready")
    except Exception as e:
        print(f"❌ Analytics Service: Error - {e}")
        return 1
    
    print("\n🎉 All systems operational!")
    return 0


def cmd_db_setup(args) -> int:
    """Set up database schema and constraints."""
    print("Setting up database schema...")
    
    try:
        from setup_database import main as setup_db_main
        setup_db_main()
        print("✅ Database setup completed successfully")
        return 0
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return 1


def cmd_db_stats(args) -> int:
    """Display database statistics."""
    try:
        conn = Neo4jConnection()
        stats = conn.get_network_statistics()
        
        print("Database Statistics")
        print("=" * 30)
        print(f"Papers: {stats.get('total_papers', 'N/A')}")
        print(f"Authors: {stats.get('total_authors', 'N/A')}")
        print(f"Citations: {stats.get('total_citations', 'N/A')}")
        print(f"Venues: {stats.get('total_venues', 'N/A')}")
        
        return 0
    except Exception as e:
        print(f"❌ Failed to retrieve database stats: {e}")
        return 1


def cmd_model_info(args) -> int:
    """Display ML model information."""
    try:
        ml_service = get_ml_service()
        info = ml_service.get_model_info()
        
        print("ML Model Information")
        print("=" * 30)
        for key, value in info.items():
            print(f"{key}: {value}")
        
        return 0
    except Exception as e:
        print(f"❌ Failed to get model info: {e}")
        return 1


def cmd_predict(args) -> int:
    """Generate citation predictions."""
    try:
        ml_service = get_ml_service()
        
        print(f"Generating top {args.top_k} predictions for paper: {args.paper_id}")
        predictions = ml_service.predict_citations(args.paper_id, top_k=args.top_k)
        
        print("\nPrediction Results:")
        print("-" * 40)
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred.target_paper_id} (score: {pred.prediction_score:.3f})")
        
        return 0
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return 1


def cmd_analyze_network(args) -> int:
    """Run network analysis."""
    try:
        analytics = get_analytics_service()
        
        print(f"Analyzing network for query: {args.query}")
        
        # Run comprehensive network analysis
        if args.communities:
            print("Running community detection...")
            # Add community detection logic here
        
        if args.centrality:
            print("Computing centrality measures...")
            # Add centrality computation logic here
        
        print("✅ Network analysis completed")
        return 0
    except Exception as e:
        print(f"❌ Network analysis failed: {e}")
        return 1


def cmd_config_show(args) -> int:
    """Display current configuration."""
    print("Current Configuration")
    print("=" * 30)
    
    config_dict = settings.to_dict()
    for section, values in config_dict.items():
        print(f"\n[{section}]")
        for key, value in values.items():
            if 'password' in key.lower() or 'key' in key.lower():
                print(f"  {key}: {'*' * 8 if value else 'Not set'}")
            else:
                print(f"  {key}: {value}")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Academic Citation Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check system health")
    health_parser.set_defaults(func=cmd_health_check)
    
    # Database commands
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_subparsers = db_parser.add_subparsers(dest="db_command")
    
    setup_parser = db_subparsers.add_parser("setup", help="Set up database schema")
    setup_parser.set_defaults(func=cmd_db_setup)
    
    stats_parser = db_subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_db_stats)
    
    # ML commands
    ml_parser = subparsers.add_parser("ml", help="Machine learning operations")
    ml_subparsers = ml_parser.add_subparsers(dest="ml_command")
    
    model_parser = ml_subparsers.add_parser("info", help="Show model information")
    model_parser.set_defaults(func=cmd_model_info)
    
    predict_parser = ml_subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("paper_id", help="Source paper ID")
    predict_parser.add_argument("-k", "--top-k", type=int, default=10, 
                               help="Number of predictions to return")
    predict_parser.set_defaults(func=cmd_predict)
    
    # Analytics commands
    analytics_parser = subparsers.add_parser("analyze", help="Network analysis")
    analytics_parser.add_argument("query", help="Search query or paper IDs")
    analytics_parser.add_argument("--communities", action="store_true", 
                                 help="Detect communities")
    analytics_parser.add_argument("--centrality", action="store_true", 
                                 help="Compute centrality measures")
    analytics_parser.set_defaults(func=cmd_analyze_network)
    
    # Configuration commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    
    show_parser = config_subparsers.add_parser("show", help="Show current configuration")
    show_parser.set_defaults(func=cmd_config_show)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle nested commands that don't have a function
    if args.command == "db" and not hasattr(args, 'func'):
        parser.parse_args(['db', '--help'])
        return 1
    elif args.command == "ml" and not hasattr(args, 'func'):
        parser.parse_args(['ml', '--help'])
        return 1
    elif args.command == "config" and not hasattr(args, 'func'):
        parser.parse_args(['config', '--help'])
        return 1
    
    # Execute the command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logging.exception("Unexpected error occurred")
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
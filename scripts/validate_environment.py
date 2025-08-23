#!/usr/bin/env python3
"""
Advanced environment validation script for Academic Citation Platform.

This script provides comprehensive environment validation beyond basic checks:
1. Validates all configuration combinations
2. Tests service connectivity and permissions
3. Checks system resources and dependencies
4. Provides detailed diagnostic information
5. Suggests optimal configuration settings
"""

import os
import sys
import platform
import psutil
import subprocess
import tempfile
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from urllib.parse import urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemRequirements:
    """System requirements for the platform."""
    min_python_version: Tuple[int, int, int] = (3, 8, 0)
    recommended_python_version: Tuple[int, int, int] = (3, 10, 0)
    min_ram_gb: int = 4
    recommended_ram_gb: int = 8
    min_disk_space_gb: float = 2.0
    required_python_packages: List[str] = None
    
    def __post_init__(self):
        if self.required_python_packages is None:
            self.required_python_packages = [
                'streamlit', 'torch', 'networkx', 'neo4j', 'pandas', 
                'numpy', 'plotly', 'pydantic', 'requests'
            ]

@dataclass
class ValidationResult:
    """Result of an individual validation check."""
    name: str
    status: str  # 'pass', 'warn', 'fail'
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None

class EnvironmentValidator:
    """Comprehensive environment validation for the Academic Citation Platform."""
    
    def __init__(self):
        self.requirements = SystemRequirements()
        self.results: List[ValidationResult] = []
        self.system_info: Dict[str, Any] = {}
        
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information."""
        info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'python': {
                'version': platform.python_version(),
                'version_info': sys.version_info,
                'executable': sys.executable,
                'path': sys.path[:3]  # First 3 entries
            },
            'memory': {
                'total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'percent_used': psutil.virtual_memory().percent
            },
            'disk': {
                'free_gb': round(psutil.disk_usage('.').free / (1024**3), 2),
                'total_gb': round(psutil.disk_usage('.').total / (1024**3), 2)
            },
            'cpu': {
                'count': psutil.cpu_count(),
                'percent': psutil.cpu_percent(interval=1)
            }
        }
        
        self.system_info = info
        return info
    
    def validate_python_version(self) -> ValidationResult:
        """Validate Python version requirements."""
        current_version = sys.version_info[:3]
        min_version = self.requirements.min_python_version
        recommended_version = self.requirements.recommended_python_version
        
        if current_version < min_version:
            return ValidationResult(
                name="Python Version",
                status="fail",
                message=f"Python {'.'.join(map(str, current_version))} is too old",
                details={
                    'current': current_version,
                    'minimum': min_version,
                    'recommended': recommended_version
                },
                recommendations=[
                    f"Upgrade to Python {'.'.join(map(str, recommended_version))} or newer",
                    "Use pyenv or conda to manage Python versions"
                ]
            )
        elif current_version < recommended_version:
            return ValidationResult(
                name="Python Version",
                status="warn", 
                message=f"Python {'.'.join(map(str, current_version))} works but newer version recommended",
                details={
                    'current': current_version,
                    'recommended': recommended_version
                },
                recommendations=[
                    f"Consider upgrading to Python {'.'.join(map(str, recommended_version))} for best performance"
                ]
            )
        else:
            return ValidationResult(
                name="Python Version",
                status="pass",
                message=f"Python {'.'.join(map(str, current_version))} is suitable",
                details={'current': current_version}
            )
    
    def validate_system_resources(self) -> List[ValidationResult]:
        """Validate system resources (RAM, disk space, etc.)."""
        results = []
        
        # Memory validation
        available_ram = self.system_info['memory']['available_gb']
        if available_ram < self.requirements.min_ram_gb:
            results.append(ValidationResult(
                name="Memory",
                status="fail",
                message=f"Insufficient RAM: {available_ram:.1f}GB available, {self.requirements.min_ram_gb}GB required",
                details=self.system_info['memory'],
                recommendations=[
                    "Close other applications to free memory",
                    "Consider adding more RAM to your system",
                    "Use lighter ML model configurations"
                ]
            ))
        elif available_ram < self.requirements.recommended_ram_gb:
            results.append(ValidationResult(
                name="Memory",
                status="warn",
                message=f"RAM acceptable but limited: {available_ram:.1f}GB available, {self.requirements.recommended_ram_gb}GB recommended",
                details=self.system_info['memory'],
                recommendations=[
                    "Consider closing other applications during ML operations",
                    f"Upgrade to {self.requirements.recommended_ram_gb}GB+ RAM for optimal performance"
                ]
            ))
        else:
            results.append(ValidationResult(
                name="Memory",
                status="pass",
                message=f"Sufficient RAM: {available_ram:.1f}GB available",
                details=self.system_info['memory']
            ))
        
        # Disk space validation
        free_disk = self.system_info['disk']['free_gb']
        if free_disk < self.requirements.min_disk_space_gb:
            results.append(ValidationResult(
                name="Disk Space",
                status="fail",
                message=f"Insufficient disk space: {free_disk:.1f}GB free, {self.requirements.min_disk_space_gb:.1f}GB required",
                details=self.system_info['disk'],
                recommendations=[
                    "Free up disk space by removing unnecessary files",
                    "Consider moving the project to a drive with more space"
                ]
            ))
        else:
            results.append(ValidationResult(
                name="Disk Space", 
                status="pass",
                message=f"Sufficient disk space: {free_disk:.1f}GB free",
                details=self.system_info['disk']
            ))
        
        return results
    
    def validate_python_packages(self) -> ValidationResult:
        """Validate required Python packages are installed."""
        missing_packages = []
        package_versions = {}
        
        for package in self.requirements.required_python_packages:
            try:
                # Try to import and get version
                if package == 'torch':
                    import torch
                    package_versions[package] = torch.__version__
                elif package == 'streamlit':
                    import streamlit
                    package_versions[package] = streamlit.__version__
                elif package == 'neo4j':
                    import neo4j
                    package_versions[package] = neo4j.__version__
                elif package == 'pandas':
                    import pandas
                    package_versions[package] = pandas.__version__
                elif package == 'numpy':
                    import numpy
                    package_versions[package] = numpy.__version__
                elif package == 'networkx':
                    import networkx
                    package_versions[package] = networkx.__version__
                elif package == 'plotly':
                    import plotly
                    package_versions[package] = plotly.__version__
                elif package == 'pydantic':
                    import pydantic
                    package_versions[package] = pydantic.__version__
                elif package == 'requests':
                    import requests
                    package_versions[package] = requests.__version__
                else:
                    # Generic import
                    __import__(package)
                    package_versions[package] = "unknown"
                    
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return ValidationResult(
                name="Python Packages",
                status="fail",
                message=f"Missing required packages: {', '.join(missing_packages)}",
                details={
                    'missing': missing_packages,
                    'installed': package_versions
                },
                recommendations=[
                    f"Install missing packages: pip install {' '.join(missing_packages)}",
                    "Or install all dependencies: pip install -e '.[all]'"
                ]
            )
        else:
            return ValidationResult(
                name="Python Packages",
                status="pass", 
                message="All required packages installed",
                details={'installed': package_versions}
            )
    
    def validate_environment_variables(self) -> ValidationResult:
        """Validate environment variable configuration."""
        required_vars = {
            'NEO4J_URI': ['NEO4J_URL'],
            'NEO4J_USER': ['NEO4J_USERNAME'], 
            'NEO4J_PASSWORD': ['NEO4J_PWD']
        }
        
        optional_vars = {
            'SEMANTIC_SCHOLAR_API_KEY': ['S2_API_KEY'],
            'LOG_LEVEL': [],
            'LOG_FILE': [],
            'CACHE_ENABLED': [],
            'ENVIRONMENT': []
        }
        
        missing_required = []
        found_vars = {}
        issues = []
        
        # Check required variables
        for primary_var, alternatives in required_vars.items():
            value = os.getenv(primary_var)
            if not value:
                # Check alternatives
                for alt_var in alternatives:
                    value = os.getenv(alt_var)
                    if value:
                        found_vars[primary_var] = f"{alt_var}={value[:10]}..."
                        break
                
                if not value:
                    missing_required.append(primary_var)
            else:
                found_vars[primary_var] = f"{primary_var}={value[:10]}..."
        
        # Check optional variables
        optional_found = {}
        for primary_var, alternatives in optional_vars.items():
            value = os.getenv(primary_var)
            if value:
                optional_found[primary_var] = f"{primary_var}={value[:10]}..."
            else:
                for alt_var in alternatives:
                    value = os.getenv(alt_var)
                    if value:
                        optional_found[primary_var] = f"{alt_var}={value[:10]}..."
                        break
        
        # Validate specific configurations
        neo4j_uri = os.getenv('NEO4J_URI') or os.getenv('NEO4J_URL')
        if neo4j_uri:
            try:
                parsed = urlparse(neo4j_uri)
                if not parsed.scheme:
                    issues.append("NEO4J_URI missing protocol (should start with neo4j:// or neo4j+s://)")
                if not parsed.hostname:
                    issues.append("NEO4J_URI missing hostname")
            except Exception as e:
                issues.append(f"Invalid NEO4J_URI format: {e}")
        
        # Generate result
        if missing_required:
            return ValidationResult(
                name="Environment Variables",
                status="fail",
                message=f"Missing required environment variables: {', '.join(missing_required)}",
                details={
                    'required_found': found_vars,
                    'optional_found': optional_found,
                    'missing_required': missing_required,
                    'issues': issues
                },
                recommendations=[
                    "Copy .env.example to .env and configure with your values",
                    "Ensure Neo4j database credentials are correct",
                    "Consider adding SEMANTIC_SCHOLAR_API_KEY for better API rate limits"
                ]
            )
        elif issues:
            return ValidationResult(
                name="Environment Variables",
                status="warn",
                message=f"Configuration issues found: {'; '.join(issues)}",
                details={
                    'required_found': found_vars,
                    'optional_found': optional_found,
                    'issues': issues
                },
                recommendations=[
                    "Review and fix environment variable formats",
                    "Test database connection with corrected settings"
                ]
            )
        else:
            recommendations = []
            if 'SEMANTIC_SCHOLAR_API_KEY' not in optional_found:
                recommendations.append("Consider adding SEMANTIC_SCHOLAR_API_KEY for better API performance")
            
            return ValidationResult(
                name="Environment Variables",
                status="pass",
                message="Environment variables properly configured",
                details={
                    'required_found': found_vars,
                    'optional_found': optional_found
                },
                recommendations=recommendations if recommendations else None
            )
    
    def validate_network_connectivity(self) -> List[ValidationResult]:
        """Validate network connectivity to external services."""
        results = []
        
        # Test Neo4j connectivity
        neo4j_uri = os.getenv('NEO4J_URI') or os.getenv('NEO4J_URL')
        if neo4j_uri:
            try:
                parsed = urlparse(neo4j_uri)
                host = parsed.hostname
                port = parsed.port or 7687
                
                # Test socket connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    results.append(ValidationResult(
                        name="Neo4j Connectivity",
                        status="pass",
                        message=f"Can connect to Neo4j at {host}:{port}",
                        details={'host': host, 'port': port}
                    ))
                else:
                    results.append(ValidationResult(
                        name="Neo4j Connectivity",
                        status="fail",
                        message=f"Cannot connect to Neo4j at {host}:{port}",
                        details={'host': host, 'port': port, 'error_code': result},
                        recommendations=[
                            "Check that Neo4j server is running",
                            "Verify firewall settings allow connection",
                            "Confirm host and port are correct"
                        ]
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    name="Neo4j Connectivity",
                    status="fail",
                    message=f"Error testing Neo4j connectivity: {e}",
                    recommendations=[
                        "Check NEO4J_URI format",
                        "Verify network connectivity"
                    ]
                ))
        
        # Test Semantic Scholar API connectivity
        try:
            import requests
            response = requests.get('https://api.semanticscholar.org/graph/v1', timeout=5)
            if response.status_code == 200:
                results.append(ValidationResult(
                    name="Semantic Scholar API",
                    status="pass",
                    message="Semantic Scholar API is accessible",
                    details={'status_code': response.status_code}
                ))
            else:
                results.append(ValidationResult(
                    name="Semantic Scholar API",
                    status="warn",
                    message=f"Semantic Scholar API returned status {response.status_code}",
                    details={'status_code': response.status_code},
                    recommendations=[
                        "API may be experiencing issues",
                        "Check https://api.semanticscholar.org/graph/v1 in browser"
                    ]
                ))
        except Exception as e:
            results.append(ValidationResult(
                name="Semantic Scholar API",
                status="warn",
                message=f"Cannot reach Semantic Scholar API: {e}",
                recommendations=[
                    "Check internet connectivity",
                    "Verify firewall allows HTTPS connections"
                ]
            ))
        
        return results
    
    def validate_file_permissions(self) -> ValidationResult:
        """Validate file system permissions for required operations."""
        test_dirs = [
            '.',  # Current directory
            'logs',  # Log directory
            'reference-codebases/citation-map-dashboard/models'  # Model directory
        ]
        
        issues = []
        details = {}
        
        for test_dir in test_dirs:
            dir_path = Path(test_dir)
            
            # Test directory access
            if dir_path.exists():
                if not os.access(dir_path, os.R_OK):
                    issues.append(f"Cannot read directory: {test_dir}")
                if not os.access(dir_path, os.W_OK):
                    issues.append(f"Cannot write to directory: {test_dir}")
                
                details[test_dir] = {
                    'exists': True,
                    'readable': os.access(dir_path, os.R_OK),
                    'writable': os.access(dir_path, os.W_OK)
                }
            else:
                # Try to create directory
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    details[test_dir] = {
                        'exists': True,
                        'readable': True,
                        'writable': True,
                        'created': True
                    }
                except Exception as e:
                    issues.append(f"Cannot create directory {test_dir}: {e}")
                    details[test_dir] = {
                        'exists': False,
                        'error': str(e)
                    }
        
        # Test file creation in current directory
        try:
            test_file = Path('test_permissions.tmp')
            test_file.write_text('test')
            test_file.unlink()
            details['file_creation'] = True
        except Exception as e:
            issues.append(f"Cannot create files: {e}")
            details['file_creation'] = False
        
        if issues:
            return ValidationResult(
                name="File Permissions",
                status="fail",
                message=f"Permission issues: {'; '.join(issues)}",
                details=details,
                recommendations=[
                    "Check directory permissions with: ls -la",
                    "Ensure user has read/write access to project directory",
                    "Consider running with appropriate user privileges"
                ]
            )
        else:
            return ValidationResult(
                name="File Permissions",
                status="pass",
                message="File system permissions are adequate",
                details=details
            )
    
    def run_validation(self) -> bool:
        """Run complete environment validation."""
        logger.info("🔍 Academic Citation Platform - Environment Validation")
        logger.info("=" * 60)
        
        # Collect system information
        logger.info("Collecting system information...")
        self.collect_system_info()
        
        # Run validation checks
        validations = [
            ("Python Version", self.validate_python_version),
            ("System Resources", self.validate_system_resources),
            ("Python Packages", self.validate_python_packages),
            ("Environment Variables", self.validate_environment_variables),
            ("Network Connectivity", self.validate_network_connectivity),
            ("File Permissions", self.validate_file_permissions)
        ]
        
        for name, validator_func in validations:
            logger.info(f"\n--- {name} ---")
            try:
                result = validator_func()
                if isinstance(result, list):
                    self.results.extend(result)
                else:
                    self.results.append(result)
            except Exception as e:
                logger.error(f"Validation error in {name}: {e}")
                self.results.append(ValidationResult(
                    name=name,
                    status="fail",
                    message=f"Validation failed with error: {e}",
                    recommendations=["Check system configuration and try again"]
                ))
        
        # Generate report
        self.generate_report()
        
        # Return overall success
        failed_critical = any(r.status == 'fail' for r in self.results)
        return not failed_critical
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        logger.info("\n" + "=" * 60)
        logger.info("🔍 ENVIRONMENT VALIDATION REPORT")
        logger.info("=" * 60)
        
        # System overview
        logger.info("\n📊 SYSTEM OVERVIEW")
        logger.info("-" * 30)
        platform_info = self.system_info['platform']
        python_info = self.system_info['python']
        memory_info = self.system_info['memory']
        
        logger.info(f"OS: {platform_info['system']} {platform_info['release']}")
        logger.info(f"Python: {python_info['version']}")
        logger.info(f"Memory: {memory_info['available_gb']:.1f}GB available / {memory_info['total_gb']:.1f}GB total")
        logger.info(f"CPU: {self.system_info['cpu']['count']} cores")
        
        # Validation results
        logger.info("\n✅ VALIDATION RESULTS")
        logger.info("-" * 30)
        
        status_icons = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        status_counts = {"pass": 0, "warn": 0, "fail": 0}
        
        for result in self.results:
            icon = status_icons[result.status]
            status_counts[result.status] += 1
            logger.info(f"{icon} {result.name}: {result.message}")
            
            # Show details for failures and warnings
            if result.status in ['fail', 'warn'] and result.details:
                logger.info(f"   Details: {result.details}")
        
        # Summary
        total_checks = len(self.results)
        logger.info(f"\n📈 SUMMARY: {status_counts['pass']}/{total_checks} passed, "
                   f"{status_counts['warn']} warnings, {status_counts['fail']} failures")
        
        # Recommendations
        all_recommendations = []
        for result in self.results:
            if result.recommendations:
                all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            logger.info("\n🚀 RECOMMENDATIONS")
            logger.info("-" * 30)
            for i, rec in enumerate(set(all_recommendations), 1):
                logger.info(f"{i}. {rec}")
        
        # Final status
        if status_counts['fail'] == 0:
            if status_counts['warn'] == 0:
                logger.info("\n🎉 Environment validation completed successfully!")
                logger.info("Your system is ready to run the Academic Citation Platform.")
            else:
                logger.info("\n✅ Environment validation completed with warnings.")
                logger.info("The platform should work, but consider addressing warnings for optimal performance.")
        else:
            logger.info("\n❌ Environment validation failed.")
            logger.info("Please address the critical issues above before running the platform.")

def main():
    """Main validation function."""
    validator = EnvironmentValidator()
    success = validator.run_validation()
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
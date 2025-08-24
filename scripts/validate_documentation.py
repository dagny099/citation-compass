#!/usr/bin/env python3
"""
Documentation validation script for the Academic Citation Platform.

This script validates the MkDocs documentation setup, checks for broken links,
validates configuration, and ensures all documentation files are properly structured.
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import urllib.parse
from urllib.parse import urlparse

def validate_mkdocs_config() -> Tuple[bool, List[str]]:
    """Validate MkDocs configuration file."""
    errors = []
    config_path = Path("mkdocs.yml")
    
    if not config_path.exists():
        errors.append("❌ mkdocs.yml not found")
        return False, errors
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['site_name', 'site_description', 'theme', 'nav']
        for field in required_fields:
            if field not in config:
                errors.append(f"❌ Missing required field: {field}")
        
        # Validate theme
        if 'theme' in config:
            if config['theme'].get('name') != 'material':
                errors.append("⚠️ Theme is not 'material' - advanced features may not work")
        
        # Validate navigation structure
        if 'nav' in config:
            nav_errors = validate_navigation(config['nav'])
            errors.extend(nav_errors)
        
        print("✅ MkDocs configuration is valid")
        
    except yaml.YAMLError as e:
        errors.append(f"❌ Invalid YAML in mkdocs.yml: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"❌ Error reading mkdocs.yml: {e}")
        return False, errors
    
    return len(errors) == 0, errors

def validate_navigation(nav: List) -> List[str]:
    """Validate navigation structure and file references."""
    errors = []
    docs_path = Path("docs")
    
    def check_nav_item(item, path=""):
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str):
                    # File reference
                    file_path = docs_path / value
                    if not file_path.exists():
                        errors.append(f"❌ Navigation references missing file: {value}")
                elif isinstance(value, list):
                    # Nested navigation
                    for subitem in value:
                        check_nav_item(subitem, f"{path}/{key}")
        elif isinstance(item, str):
            # Direct file reference
            file_path = docs_path / item
            if not file_path.exists():
                errors.append(f"❌ Navigation references missing file: {item}")
    
    for item in nav:
        check_nav_item(item)
    
    return errors

def validate_documentation_structure() -> Tuple[bool, List[str]]:
    """Validate documentation directory structure."""
    errors = []
    docs_path = Path("docs")
    
    if not docs_path.exists():
        errors.append("❌ docs/ directory not found")
        return False, errors
    
    # Check required directories
    required_dirs = [
        "getting-started",
        "user-guide", 
        "developer-guide",
        "notebooks",
        "api",
        "assets",
        "overrides",
        "includes"
    ]
    
    for dir_name in required_dirs:
        dir_path = docs_path / dir_name
        if not dir_path.exists():
            errors.append(f"❌ Required directory missing: docs/{dir_name}")
    
    # Check required files
    required_files = [
        "index.md",
        "getting-started/installation.md",
        "getting-started/configuration.md",
        "getting-started/quick-start.md",
        "user-guide/overview.md",
        "developer-guide/architecture.md",
        "api/services.md",
        "notebooks/overview.md"
    ]
    
    for file_name in required_files:
        file_path = docs_path / file_name
        if not file_path.exists():
            errors.append(f"❌ Required file missing: docs/{file_name}")
        else:
            # Check if file has content
            if file_path.stat().st_size == 0:
                errors.append(f"⚠️ File is empty: docs/{file_name}")
    
    if len(errors) == 0:
        print("✅ Documentation structure is valid")
    
    return len(errors) == 0, errors

def validate_markdown_files() -> Tuple[bool, List[str]]:
    """Validate markdown files for basic structure and content."""
    errors = []
    docs_path = Path("docs")
    
    markdown_files = list(docs_path.rglob("*.md"))
    
    for md_file in markdown_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has a title (H1 heading)
            if not content.strip().startswith('#'):
                errors.append(f"⚠️ File lacks H1 heading: {md_file.relative_to(docs_path)}")
            
            # Check for basic content length
            if len(content.strip()) < 100:
                errors.append(f"⚠️ File may be too short: {md_file.relative_to(docs_path)}")
            
        except UnicodeDecodeError:
            errors.append(f"❌ File encoding error: {md_file.relative_to(docs_path)}")
        except Exception as e:
            errors.append(f"❌ Error reading file {md_file.relative_to(docs_path)}: {e}")
    
    print(f"✅ Validated {len(markdown_files)} markdown files")
    return len(errors) == 0, errors

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed."""
    errors = []
    
    required_packages = [
        "mkdocs",
        "mkdocs-material",
        "mkdocstrings[python]",
        "pymdown-extensions"
    ]
    
    for package in required_packages:
        try:
            result = subprocess.run([
                sys.executable, "-c", f"import {package.replace('-', '_').split('[')[0]}"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                errors.append(f"❌ Missing package: {package}")
        except Exception:
            errors.append(f"❌ Cannot check package: {package}")
    
    # Check MkDocs version
    try:
        result = subprocess.run(["mkdocs", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ MkDocs version: {result.stdout.strip()}")
        else:
            errors.append("❌ MkDocs not properly installed")
    except FileNotFoundError:
        errors.append("❌ MkDocs command not found")
    
    return len(errors) == 0, errors

def test_build() -> Tuple[bool, List[str]]:
    """Test building the documentation."""
    errors = []
    
    try:
        # Test build in strict mode
        result = subprocess.run([
            "mkdocs", "build", "--strict", "--clean"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Documentation builds successfully")
            
            # Check if site directory was created
            site_path = Path("site")
            if site_path.exists():
                index_file = site_path / "index.html"
                if index_file.exists():
                    print("✅ Generated HTML files successfully")
                else:
                    errors.append("❌ index.html not generated")
            else:
                errors.append("❌ site/ directory not created")
                
        else:
            errors.append(f"❌ Build failed: {result.stderr}")
            
    except FileNotFoundError:
        errors.append("❌ MkDocs command not found")
    except Exception as e:
        errors.append(f"❌ Build test error: {e}")
    
    return len(errors) == 0, errors

def generate_report(results: Dict[str, Tuple[bool, List[str]]]) -> None:
    """Generate validation report."""
    print("\n" + "="*60)
    print("📋 DOCUMENTATION VALIDATION REPORT")
    print("="*60)
    
    total_errors = 0
    total_warnings = 0
    
    for check_name, (success, messages) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{check_name}: {status}")
        
        if messages:
            for message in messages:
                print(f"  {message}")
                if "❌" in message:
                    total_errors += 1
                elif "⚠️" in message:
                    total_warnings += 1
    
    print(f"\n📊 Summary:")
    print(f"  Total Checks: {len(results)}")
    print(f"  Passed: {sum(1 for success, _ in results.values() if success)}")
    print(f"  Failed: {sum(1 for success, _ in results.values() if not success)}")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")
    
    if total_errors == 0:
        print("\n🎉 Documentation validation completed successfully!")
        print("   Ready to deploy with: mkdocs serve")
    else:
        print(f"\n⚠️  Please fix {total_errors} errors before deployment")
        sys.exit(1)

def main():
    """Run all validation checks."""
    print("🔍 Academic Citation Platform - Documentation Validation")
    print("="*60)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Run validation checks
    results = {
        "Dependencies Check": check_dependencies(),
        "MkDocs Configuration": validate_mkdocs_config(),
        "Documentation Structure": validate_documentation_structure(),
        "Markdown Files": validate_markdown_files(),
        "Build Test": test_build()
    }
    
    # Generate report
    generate_report(results)

if __name__ == "__main__":
    main()
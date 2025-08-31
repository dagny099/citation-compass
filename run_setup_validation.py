#!/usr/bin/env python3
"""
Setup validation script that runs all immediate actions to verify the platform is ready.

This script:
1. Validates environment configuration
2. Tests database setup (if configured)
3. Verifies ML models are accessible  
4. Checks documentation structure
5. Runs basic functionality tests
6. Provides setup recommendations
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SetupValidator:
    """Comprehensive setup validation for Academic Citation Platform."""
    
    def __init__(self):
        self.results = {}
        self.recommendations = []
    
    def validate_environment(self) -> bool:
        """Check environment configuration."""
        logger.info("🔧 Validating environment configuration...")
        
        required_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD']
        missing_vars = []
        
        for var in required_vars:
            # Check multiple naming conventions
            alt_vars = {
                'NEO4J_URI': ['NEO4J_URL'],
                'NEO4J_USER': ['NEO4J_USERNAME'],
                'NEO4J_PASSWORD': ['NEO4J_PWD']
            }
            
            value = os.getenv(var)
            if not value:
                # Check alternatives
                for alt_var in alt_vars.get(var, []):
                    value = os.getenv(alt_var)
                    if value:
                        break
            
            if not value:
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {missing_vars}")
            self.recommendations.append("Copy .env.example to .env and configure database settings")
            self.results['environment'] = False
            return False
        
        logger.info("✅ Environment variables configured")
        self.results['environment'] = True
        return True
    
    def validate_database_setup(self) -> bool:
        """Test database connectivity and setup."""
        logger.info("🗄️  Validating database setup...")
        
        try:
            # Run database setup script
            result = subprocess.run(
                [sys.executable, 'setup_database.py'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ Database setup successful")
                self.results['database'] = True
                return True
            else:
                logger.error(f"❌ Database setup failed: {result.stderr}")
                self.recommendations.append("Check Neo4j server is running and credentials are correct")
                self.results['database'] = False
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Database setup timed out")
            self.recommendations.append("Check network connectivity to Neo4j server")
            self.results['database'] = False
            return False
        except FileNotFoundError:
            logger.error("❌ Database setup script not found")
            self.results['database'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Database setup error: {e}")
            self.results['database'] = False
            return False
    
    def validate_ml_models(self) -> bool:
        """Verify ML model files and functionality."""
        logger.info("🤖 Validating ML models...")
        
        try:
            result = subprocess.run(
                [sys.executable, 'verify_models.py'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✅ ML models verified successfully")
                self.results['ml_models'] = True
                return True
            else:
                logger.error(f"❌ ML model verification failed: {result.stderr}")
                self.recommendations.append("Check ML model files in models/ directory")
                self.results['ml_models'] = False
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ ML model verification timed out")
            self.recommendations.append("Ensure sufficient RAM (4GB+) for model loading")
            self.results['ml_models'] = False
            return False
        except Exception as e:
            logger.error(f"❌ ML model verification error: {e}")
            self.results['ml_models'] = False
            return False
    
    def validate_documentation(self) -> bool:
        """Check documentation structure."""
        logger.info("📚 Validating documentation...")
        
        required_docs = [
            'docs/architecture.md',
            'docs/api.md', 
            'docs/setup.md'
        ]
        
        missing_docs = []
        for doc_path in required_docs:
            if not Path(doc_path).exists():
                missing_docs.append(doc_path)
        
        if missing_docs:
            logger.error(f"❌ Missing documentation: {missing_docs}")
            self.results['documentation'] = False
            return False
        
        logger.info("✅ Documentation structure complete")
        self.results['documentation'] = True
        return True
    
    def validate_basic_tests(self) -> bool:
        """Run basic functionality tests."""
        logger.info("🧪 Running basic functionality tests...")
        
        try:
            # Run a subset of tests that don't require external dependencies
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/test_models_simple.py', '-v'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ Basic tests passed")
                self.results['basic_tests'] = True
                return True
            else:
                logger.warning(f"⚠️  Some tests failed: {result.stderr}")
                self.recommendations.append("Review test failures and fix dependencies")
                self.results['basic_tests'] = False
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Tests timed out")
            self.results['basic_tests'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Test execution error: {e}")
            self.results['basic_tests'] = False
            return False
    
    def validate_streamlit_app(self) -> bool:
        """Check Streamlit app can be imported."""
        logger.info("🌐 Validating Streamlit application...")
        
        try:
            # Test app import without running
            result = subprocess.run(
                [sys.executable, '-c', 'import app; print("Streamlit app import successful")'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("✅ Streamlit app imports successfully")
                self.results['streamlit_app'] = True
                return True
            else:
                logger.error(f"❌ Streamlit app import failed: {result.stderr}")
                self.recommendations.append("Check Streamlit dependencies and page imports")
                self.results['streamlit_app'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Streamlit validation error: {e}")
            self.results['streamlit_app'] = False
            return False
    
    def generate_summary(self) -> None:
        """Generate setup validation summary."""
        logger.info("📋 Setup Validation Summary")
        logger.info("=" * 50)
        
        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result)
        
        logger.info(f"Overall Status: {passed_checks}/{total_checks} checks passed")
        logger.info("")
        
        # Individual results
        status_icons = {True: "✅", False: "❌"}
        for check, result in self.results.items():
            icon = status_icons[result]
            logger.info(f"{icon} {check.replace('_', ' ').title()}: {'PASS' if result else 'FAIL'}")
        
        logger.info("")
        
        # Recommendations
        if self.recommendations:
            logger.info("🚀 Recommendations:")
            for i, recommendation in enumerate(self.recommendations, 1):
                logger.info(f"{i}. {recommendation}")
        
        logger.info("")
        
        # Next steps
        if passed_checks == total_checks:
            logger.info("🎉 Setup validation completed successfully!")
            logger.info("You can now run: streamlit run app.py")
        else:
            logger.info("⚠️  Setup issues detected. Please address recommendations above.")
            logger.info("Run this script again after making fixes.")
    
    def run_validation(self) -> bool:
        """Run complete validation suite."""
        logger.info("🚀 Academic Citation Platform - Setup Validation")
        logger.info("=" * 60)
        
        # Run all validation checks
        checks = [
            ('Environment Configuration', self.validate_environment),
            ('Database Setup', self.validate_database_setup),
            ('ML Models', self.validate_ml_models),
            ('Documentation', self.validate_documentation),
            ('Basic Tests', self.validate_basic_tests),
            ('Streamlit App', self.validate_streamlit_app)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            logger.info(f"\n--- {check_name} ---")
            try:
                result = check_func()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {check_name} validation failed with exception: {e}")
                self.results[check_name.lower().replace(' ', '_')] = False
                all_passed = False
        
        # Generate summary
        logger.info("")
        self.generate_summary()
        
        return all_passed

def main():
    """Main validation function."""
    validator = SetupValidator()
    success = validator.run_validation()
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
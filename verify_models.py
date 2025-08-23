#!/usr/bin/env python3
"""
Model verification script for Academic Citation Platform.

This script:
1. Checks if ML model files exist and are accessible
2. Loads the TransE model and validates its structure
3. Tests model predictions with sample data
4. Verifies entity mappings are valid
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_model_files() -> Dict[str, Path]:
    """Check if all required model files exist."""
    base_path = Path("reference-codebases/citation-map-dashboard/models")
    
    required_files = {
        'model': base_path / 'transe_citation_model.pt',
        'entity_mapping': base_path / 'entity_mapping.pkl', 
        'metadata': base_path / 'training_metadata.pkl'
    }
    
    logger.info("Checking model files...")
    
    missing_files = []
    file_info = {}
    
    for name, path in required_files.items():
        if path.exists():
            file_size = path.stat().st_size
            file_info[name] = {
                'path': path,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'exists': True
            }
            logger.info(f"✅ {name}: {path} ({file_info[name]['size_mb']} MB)")
        else:
            missing_files.append(name)
            file_info[name] = {'path': path, 'exists': False}
            logger.error(f"❌ {name}: {path} (missing)")
    
    if missing_files:
        logger.error(f"Missing model files: {missing_files}")
        return {}
    
    logger.info("✅ All model files found")
    return {name: info['path'] for name, info in file_info.items()}

def verify_entity_mapping(mapping_path: Path) -> Optional[Dict]:
    """Load and verify entity mapping file."""
    logger.info("Verifying entity mapping...")
    
    try:
        with open(mapping_path, 'rb') as f:
            entity_mapping = pickle.load(f)
        
        if not isinstance(entity_mapping, dict):
            logger.error(f"❌ Entity mapping is not a dictionary: {type(entity_mapping)}")
            return None
        
        num_entities = len(entity_mapping)
        sample_keys = list(entity_mapping.keys())[:3]
        sample_values = [entity_mapping[k] for k in sample_keys]
        
        logger.info(f"✅ Entity mapping loaded: {num_entities} entities")
        logger.info(f"   Sample mappings: {dict(zip(sample_keys, sample_values))}")
        
        # Verify mapping values are integers
        if not all(isinstance(v, int) for v in entity_mapping.values()):
            logger.error("❌ Entity mapping values are not all integers")
            return None
        
        # Check for expected paper ID format (Semantic Scholar IDs)
        semantic_scholar_ids = [k for k in sample_keys if len(k) == 40 and k.isalnum()]
        if semantic_scholar_ids:
            logger.info(f"✅ Found Semantic Scholar paper IDs: {len(semantic_scholar_ids)} samples")
        else:
            logger.warning("⚠️  No Semantic Scholar paper IDs detected in sample")
        
        return entity_mapping
        
    except Exception as e:
        logger.error(f"❌ Failed to load entity mapping: {e}")
        return None

def verify_metadata(metadata_path: Path) -> Optional[Dict]:
    """Load and verify training metadata."""
    logger.info("Verifying training metadata...")
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"✅ Metadata loaded: {type(metadata)}")
        
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                logger.info(f"   {key}: {value}")
        else:
            logger.info(f"   Content: {metadata}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"❌ Failed to load metadata: {e}")
        return None

def verify_model(model_path: Path, entity_mapping: Dict) -> bool:
    """Load and verify the TransE model."""
    logger.info("Verifying TransE model...")
    
    try:
        # Load model checkpoint
        device = 'cpu'  # Use CPU for verification
        checkpoint = torch.load(model_path, map_location=device)
        
        logger.info(f"✅ Model checkpoint loaded")
        logger.info(f"   Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check for expected checkpoint structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info(f"   Model state dict keys: {list(state_dict.keys())}")
            
            # Check embedding dimensions
            if 'entity_embeddings.weight' in state_dict:
                entity_emb_shape = state_dict['entity_embeddings.weight'].shape
                logger.info(f"   Entity embeddings shape: {entity_emb_shape}")
                
                expected_entities = len(entity_mapping)
                actual_entities = entity_emb_shape[0]
                
                if actual_entities == expected_entities:
                    logger.info(f"✅ Entity count matches: {actual_entities}")
                else:
                    logger.warning(f"⚠️  Entity count mismatch: model={actual_entities}, mapping={expected_entities}")
                
                embedding_dim = entity_emb_shape[1]
                logger.info(f"   Embedding dimension: {embedding_dim}")
            
            if 'relation_embedding.weight' in state_dict:
                rel_emb_shape = state_dict['relation_embedding.weight'].shape
                logger.info(f"   Relation embeddings shape: {rel_emb_shape}")
        
        # Test model loading with our service
        try:
            from src.services.ml_service import TransEModel
            
            # Get model configuration
            if "model_config" in checkpoint:
                config = checkpoint["model_config"]
                logger.info(f"   Model config: {config}")
            else:
                config = {
                    "num_entities": len(entity_mapping),
                    "embedding_dim": 128,
                    "margin": 1.0,
                    "p_norm": 1
                }
                logger.info(f"   Using default config: {config}")
            
            # Create model instance
            model = TransEModel(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            logger.info("✅ Model loaded successfully with TransE service")
            
            # Test forward pass
            test_sources = torch.tensor([0, 1])
            test_targets = torch.tensor([1, 2])
            
            with torch.no_grad():
                scores = model(test_sources, test_targets)
                logger.info(f"   Test prediction scores: {scores.tolist()}")
            
            logger.info("✅ Model forward pass successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model service loading failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def test_ml_service() -> bool:
    """Test the full ML service pipeline."""
    logger.info("Testing ML service pipeline...")
    
    try:
        from src.services.ml_service import get_ml_service
        
        # Initialize service
        ml_service = get_ml_service()
        
        # Health check
        health = ml_service.health_check()
        logger.info(f"   Health check: {health.get('status', 'unknown')}")
        
        if health.get('status') == 'healthy':
            logger.info(f"   Model loaded: {health.get('model_loaded', False)}")
            logger.info(f"   Entities: {health.get('num_entities', 0)}")
            logger.info(f"   Device: {health.get('device', 'unknown')}")
            
            # Test model info
            model_info = ml_service.get_model_info()
            logger.info(f"   Model info: {model_info.model_name} ({model_info.model_type.value})")
            
            # Test with a sample paper ID if we have entities
            if ml_service.entity_mapping:
                sample_paper_id = next(iter(ml_service.entity_mapping.keys()))
                logger.info(f"   Testing with sample paper: {sample_paper_id}")
                
                # Test embedding retrieval
                embedding = ml_service.get_paper_embedding(sample_paper_id)
                if embedding:
                    logger.info(f"   ✅ Retrieved embedding: dim={embedding.embedding_dim}")
                
                # Test predictions
                predictions = ml_service.predict_citations(sample_paper_id, top_k=3)
                if predictions:
                    logger.info(f"   ✅ Generated {len(predictions)} predictions")
                    for i, pred in enumerate(predictions[:2]):
                        logger.info(f"      {i+1}. {pred.target_paper_id} (score: {pred.prediction_score:.3f})")
                else:
                    logger.warning("   ⚠️  No predictions generated")
            
            logger.info("✅ ML service test successful")
            return True
        else:
            logger.error(f"❌ ML service health check failed: {health}")
            return False
            
    except Exception as e:
        logger.error(f"❌ ML service test failed: {e}")
        return False

def main():
    """Main verification function."""
    logger.info("🚀 Academic Citation Platform - Model Verification")
    logger.info("=" * 60)
    
    # Check model files
    model_paths = check_model_files()
    if not model_paths:
        logger.error("❌ Verification failed - missing model files")
        return False
    
    # Verify entity mapping
    entity_mapping = verify_entity_mapping(model_paths['entity_mapping'])
    if not entity_mapping:
        logger.error("❌ Verification failed - invalid entity mapping")
        return False
    
    # Verify metadata
    metadata = verify_metadata(model_paths['metadata'])
    if metadata is None:
        logger.warning("⚠️  Could not load metadata (non-critical)")
    
    # Verify model
    if not verify_model(model_paths['model'], entity_mapping):
        logger.error("❌ Verification failed - model loading issues")
        return False
    
    # Test ML service
    if not test_ml_service():
        logger.error("❌ Verification failed - ML service issues")
        return False
    
    logger.info("🎉 Model verification completed successfully!")
    logger.info("All ML components are ready for use.")
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
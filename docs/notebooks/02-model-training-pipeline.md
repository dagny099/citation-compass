# TransE Model Training Pipeline

The **TransE Model Training Pipeline** notebook implements and trains the TransE (Translating Embeddings) model for citation link prediction. This notebook transforms citation network analysis into a machine learning model capable of predicting missing academic connections.

## 🎯 Learning Objectives

By completing this notebook, you will:

- **Master TransE architecture** for knowledge graph embedding
- **Understand citation prediction** as a link prediction problem
- **Learn training strategies** for sparse academic networks
- **Implement margin ranking loss** for contrastive learning
- **Develop model evaluation** and validation techniques
- **Create production-ready models** with proper persistence

## 📋 Prerequisites

### Required Knowledge
- Understanding of neural networks and embeddings
- Familiarity with PyTorch framework
- Knowledge of citation networks (from notebook 01)
- Experience with train/test splits and validation

### System Requirements
- PyTorch installation with CUDA support (recommended)
- Sufficient GPU/CPU resources for training
- 8GB+ RAM recommended for medium networks
- Storage space for model checkpoints and embeddings

### Data Prerequisites
- **Completed Notebook 01**: Network analysis results
- **Citation network data**: Papers and citation relationships  
- **Entity mappings**: Paper ID to index conversions
- **Export files**: `exploration_data.pkl` from previous analysis

## 🧠 TransE Model Overview

TransE learns vector representations where citation relationships follow a simple translation principle:

```
embedding(citing_paper) + embedding("CITES") ≈ embedding(cited_paper)
```

### Core Architecture
- **Entity Embeddings**: Each paper gets a dense vector representation
- **Relation Embedding**: Single "CITES" relation vector
- **Translation Principle**: Vector arithmetic captures semantic relationships
- **Margin Ranking Loss**: Positive citations score lower than negative ones

### Why TransE for Citations?
- **Semantic Learning**: Captures implicit relationships between papers
- **Scalability**: Efficient for sparse networks with millions of entities
- **Interpretability**: Embedding space reveals academic communities
- **Transfer Learning**: Embeddings useful for multiple downstream tasks

## 🚀 Quick Start Guide

### Option 1: Complete Training Pipeline
```python
# Launch the notebook
jupyter notebook notebooks/02_model_training_pipeline.ipynb

# Execute the full pipeline:
# 1. Load data from exploration notebook
# 2. Prepare train/test splits with negative sampling
# 3. Initialize TransE model with optimal hyperparameters
# 4. Train with margin ranking loss and early stopping
# 5. Monitor training progress and embedding quality
# 6. Save trained model with comprehensive metadata
```

### Option 2: Custom Configuration
```python
# Modify key parameters for your use case:
MODEL_CONFIG = {
    'embedding_dim': 128,        # Adjust based on network size
    'learning_rate': 0.01,       # Tune for convergence speed
    'margin': 1.0,              # Margin for ranking loss
    'batch_size': 1024          # Balance memory and training speed
}
```

## 📊 Step-by-Step Training Workflow

### Step 1: Data Loading and Validation
**Purpose**: Load comprehensive data from exploration phase and validate readiness

**Key Activities**:
- Import exploration results (`exploration_data.pkl`)
- Validate entity mappings and citation relationships
- Verify data integrity and completeness
- Set up device configuration (GPU/CPU)

**Expected Output**:
```
📚 Loading data from comprehensive exploration...
✅ Loaded exploration data from previous analysis
   Papers: 12,553
   Citations: 18,912
   Entity mapping: 12,553 entities
🎯 Data ready for model training pipeline!
```

**Data Validation Checks**:
- Entity mapping completeness
- Citation edge validity
- Memory requirements estimation
- GPU availability assessment

### Step 2: Train/Test Split and Negative Sampling
**Purpose**: Create balanced datasets with proper train/test separation

**Key Activities**:
- Split positive citations (80/20 train/test)
- Generate negative samples using random sampling
- Ensure no test set leakage into training
- Balance positive/negative sample ratios

**Expected Output**:
```
📊 Train/Test Split:
   Training positive edges: 15,129
   Training negative edges: 15,129
   Test positive edges: 3,783
   Test negative edges: 3,783
   Total training samples: 30,258
```

**Negative Sampling Strategy**:
- **Random Sampling**: Efficient for large networks
- **Exclusion Logic**: Prevent existing citations as negatives
- **Balanced Ratios**: Equal positive/negative samples
- **Quality Control**: Avoid self-loops and duplicates

### Step 3: Model Architecture Design
**Purpose**: Initialize TransE model with optimal configuration for citation networks

**Key Activities**:
- Design entity and relation embedding layers
- Implement TransE scoring function
- Configure margin ranking loss
- Initialize embeddings with proper scaling

**Model Architecture**:
```python
class TransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, norm_p=1):
        # Entity embeddings: [num_papers x embedding_dim]
        # Relation embeddings: [1 x embedding_dim] for "CITES"
        # Scoring: ||head + relation - tail||_p
```

**Expected Output**:
```
✅ TransE model created successfully!
   Model device: cuda:0
   Actual parameters: 1,607,024
   Architecture: TransE(12553, 1, 128)
```

**Architecture Decisions**:
- **Embedding Dimension**: Balance expressiveness vs. memory
- **Norm Type**: L1 or L2 distance for scoring
- **Initialization**: Uniform distribution with proper scaling
- **Regularization**: L2 weight decay for generalization

### Step 4: Training Configuration and Optimization
**Purpose**: Set up training loop with monitoring and early stopping

**Training Configuration**:
```python
TRAINING_CONFIG = {
    'epochs': 100,              # Maximum training epochs
    'batch_size': 1024,         # Balance memory and convergence
    'eval_frequency': 10,       # Validation every N epochs
    'early_stopping_patience': 20,  # Stop if no improvement
    'normalize_frequency': 10   # Embedding normalization
}
```

**Optimization Strategy**:
- **Adam Optimizer**: Adaptive learning rates
- **Learning Rate**: 0.01 with potential decay
- **L2 Regularization**: Prevent overfitting
- **Embedding Normalization**: Maintain unit norm constraints

### Step 5: Training Loop Execution
**Purpose**: Execute main training with comprehensive monitoring

**Key Activities**:
- Batch-wise margin ranking loss computation
- Gradient descent parameter updates
- Periodic embedding normalization
- Training progress tracking and visualization

**Training Progress Monitoring**:
```
🚂 Training Progress:
Epoch 50/100: Loss: 0.1234 (best: 0.1156) | Time: 12.3s | Batches: 30
✅ Model converged successfully (loss variance: 0.000045)
```

**Loss Function**:
```python
loss = max(0, positive_score - negative_score + margin)
```
- **Lower scores** = more plausible citations
- **Margin separation** between positive and negative examples
- **Convergence** when loss stabilizes at minimum

### Step 6: Training Validation and Quality Assessment
**Purpose**: Validate model learning through score analysis

**Validation Metrics**:
- **Score Separation**: Difference between positive and negative means
- **Ranking Accuracy**: Percentage where negatives score higher than positives
- **Embedding Quality**: Norm consistency and value distribution
- **Convergence Assessment**: Loss stability and improvement trends

**Expected Validation Results**:
```
📊 Training Sample Validation (1000 samples):
   Positive citations mean: 8.2341 ± 2.1456
   Negative citations mean: 12.5678 ± 3.2145
   Score separation: 4.3337
   Ranking accuracy: 0.847 (84.7%)
✅ Excellent: Model learned to distinguish citations well
```

### Step 7: Embedding Analysis and Visualization
**Purpose**: Understand what the model learned about paper relationships

**Analysis Components**:
- **Embedding Statistics**: Norms, value ranges, distributions
- **Similarity Analysis**: Cosine similarities between papers
- **Quality Metrics**: Consistency and interpretability measures
- **Visualization**: t-SNE plots of learned embeddings

**Embedding Quality Indicators**:
```
📐 Embedding Dimensions and Properties:
   Entity embeddings shape: [12553, 128]
   Mean norm: 1.0034 ± 0.0123 ✅ Good norm consistency
   Value range: [-2.34, 2.45] ✅ Reasonable value range
   Avg similarity: 0.0234 ✅ Good similarity distribution
```

### Step 8: Model Persistence and Metadata
**Purpose**: Save complete model state for deployment and evaluation

**Saved Artifacts**:
```
💾 Model Save Summary:
   📦 Complete model: transe_citation_model.pt
   🗺️ Entity mapping: entity_mapping.pkl
   📊 Training metadata: training_metadata.json
   🧪 Test data: test_data.pkl
   📖 Instructions: model_loading_instructions.txt
```

**Model Checkpoint Contents**:
- **Model State**: Trained parameters and architecture
- **Training History**: Loss curves and convergence metrics
- **Configuration**: All hyperparameters and settings
- **Metadata**: Timestamps, versions, system information

## 🔧 Advanced Configuration

### Hyperparameter Tuning

#### Embedding Dimension Selection
```python
# Rule of thumb for embedding dimensions
embedding_dim = min(256, max(64, num_entities // 100))

# For different network sizes:
# Small (<5K papers): 64-128 dimensions
# Medium (5K-50K papers): 128-256 dimensions  
# Large (>50K papers): 256-512 dimensions
```

#### Learning Rate Scheduling
```python
# Adaptive learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

#### Batch Size Optimization
```python
# Memory-based batch size calculation
max_batch_size = min(2048, available_memory_gb * 256)
optimal_batch_size = min(max_batch_size, len(training_edges) // 100)
```

### Training Strategies for Large Networks

#### Gradient Accumulation
```python
# For memory-constrained environments
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps
```

#### Mixed Precision Training
```python
# Faster training with FP16
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = compute_loss(model_output)
```

## 🚨 Troubleshooting Guide

### Common Training Issues

#### Slow Convergence
**Symptoms**: Loss decreases very slowly or plateaus early
**Solutions**:
- Increase learning rate (try 0.02-0.05)
- Reduce batch size for more frequent updates
- Check negative sampling quality
- Adjust margin parameter (try 0.5-2.0)

#### Memory Errors
**Symptoms**: CUDA out of memory or system RAM exhaustion
**Solutions**:
- Reduce batch size to 256-512
- Use gradient accumulation
- Enable mixed precision training
- Move to CPU training if necessary

#### Training Instability
**Symptoms**: Loss fluctuates wildly or increases
**Solutions**:
- Reduce learning rate (try 0.001-0.005)
- Add gradient clipping
- Check embedding initialization
- Increase L2 regularization

#### Poor Validation Performance
**Symptoms**: Low ranking accuracy or poor score separation
**Solutions**:
- Increase training epochs
- Improve negative sampling strategy
- Adjust embedding dimension
- Check for data leakage

### Model Quality Issues

#### Embedding Collapse
**Symptoms**: All embeddings become similar
**Solutions**:
- Reduce learning rate
- Increase margin parameter
- Add embedding normalization
- Check initialization scaling

#### Overfitting Detection
**Symptoms**: Training loss decreases but validation performance drops
**Solutions**:
- Add L2 regularization
- Reduce model complexity
- Increase training data
- Implement early stopping

## 📈 Performance Optimization

### GPU Acceleration
- **CUDA**: Significant speedup for large networks
- **Batch Processing**: Vectorized operations
- **Memory Management**: Efficient tensor allocation
- **Mixed Precision**: FP16 training for faster computation

### Scalability Considerations
- **Memory Scaling**: O(num_entities × embedding_dim)
- **Compute Scaling**: O(num_citations × batch_size)
- **Storage Requirements**: Model size and checkpoint management
- **Distributed Training**: For very large networks

## 🔗 Integration and Next Steps

### Data Pipeline Integration
This notebook integrates seamlessly with:
- **Previous Analysis**: Uses exploration results from notebook 01
- **Next Evaluation**: Exports model for evaluation in notebook 03
- **Production Deployment**: Generates deployment-ready model files

### Model Deployment Preparation
The trained model is ready for:
- **Citation Prediction**: Link prediction on new papers
- **Recommendation Systems**: Similar paper discovery
- **Embedding Analysis**: Paper clustering and visualization
- **Research Tools**: Integration with academic platforms

### Quality Assurance Checklist
Before proceeding to evaluation:
- [ ] Training converged successfully (stable loss)
- [ ] Validation metrics show learning (score separation > 0.5)
- [ ] Embedding quality metrics are healthy
- [ ] Model files saved with complete metadata
- [ ] Test data preserved for evaluation

## 🌟 Best Practices

### Training Methodology
1. **Start Small**: Test on subset before full training
2. **Monitor Closely**: Watch for convergence and overfitting
3. **Validate Frequently**: Check model learning throughout training
4. **Save Checkpoints**: Regular saves prevent loss of progress
5. **Document Everything**: Record hyperparameters and results

### Code Quality
1. **Reproducible Seeds**: Set random seeds for consistent results
2. **Error Handling**: Graceful failure for edge cases
3. **Memory Management**: Clear cached tensors
4. **Progress Tracking**: Use tqdm for long operations
5. **Logging**: Comprehensive training logs

### Model Management
1. **Version Control**: Tag model versions with performance
2. **Metadata**: Complete documentation of training process
3. **Backup Strategy**: Multiple checkpoint saves
4. **Testing**: Validate model loading and inference
5. **Documentation**: Clear usage instructions

## 📚 Research Applications

### Citation Prediction
- **Missing Citations**: Discover overlooked references
- **Literature Review**: Systematic paper discovery
- **Research Gaps**: Identify underexplored connections

### Network Analysis
- **Paper Clustering**: Group semantically similar works
- **Field Evolution**: Track research area development
- **Collaboration Networks**: Find potential research partnerships

### Academic Tools
- **Smart Libraries**: Improved recommendation systems
- **Research Assistants**: AI-powered literature discovery
- **Peer Review**: Automated relevance assessment

## ➡️ Continue to Evaluation

Upon successful completion of this notebook:

1. **Verify Training**: Check all validation metrics are satisfactory
2. **Inspect Artifacts**: Review saved model files and metadata  
3. **Prepare for Evaluation**: Ensure test data is properly separated
4. **Document Results**: Record training insights and performance

**Next Step**: [Prediction Evaluation](03-prediction-evaluation.md) to comprehensively assess your trained TransE model performance.

---

*This notebook transforms citation network analysis into actionable AI models. The TransE embeddings learned here form the foundation for intelligent academic discovery systems.*
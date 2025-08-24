## **Getting Started with Link Prediction**

#### **Step 1: Preprocess the Data**
Ensure your data is in a graph format. Use triples such as:
- `("Paper A", "cites", "Paper B")`
- `("Paper A", "authored by", "Author X")`

If stored in a Pandas DataFrame:
```python
import pandas as pd

# Example triples
triples = pd.DataFrame([
    {"subject": "Paper A", "predicate": "cites", "object": "Paper B"},
    {"subject": "Paper B", "predicate": "cites", "object": "Paper C"},
    {"subject": "Paper A", "predicate": "authored by", "object": "Author X"},
])

print(triples.head())
```

#### **Step 2: Split Data for Training and Testing**
Split the edges (triples) into training and testing sets:
- Training set: 80% of the triples.
- Testing set: 20%, including some negative samples.

**Example:**
```python
from sklearn.model_selection import train_test_split

# Split triples
train_triples, test_triples = train_test_split(triples, test_size=0.2, random_state=42)
```

#### **Step 3: Create Graph Embeddings**
Use **PyTorch** with a library like `torch-geometric` or `pykeen` for embeddings.

- **TransE (Translation Embeddings):** Projects entities into a vector space and learns relations as translations.

Example with `pykeen`:
```python
from pykeen.pipeline import pipeline

# Create a pipeline for TransE
result = pipeline(
    model="TransE",
    dataset="your_dataset",  # Replace with your dataset loader
    training=train_triples,
    testing=test_triples,
)

# Access embeddings
entity_embeddings = result.model.entity_embeddings()
```

---

#### **Step 4: Perform Link Prediction**
Predict missing edges (e.g., potential citations).

Example:
- Given `("Paper X", "cites", ?)`, predict the missing paper.
- Compute a ranking of likely links for evaluation.

---

#### **Step 5: Evaluate the Model**
Use metrics like:
- **Mean Rank (MR):** Average rank of true edges among all predictions.
- **Hits@K:** Fraction of true edges in the top \(K\) predictions.

**Example Evaluation in PyTorch:**
```python
from pykeen.evaluation import RankBasedEvaluator

# Evaluate predictions
evaluator = RankBasedEvaluator()
results = evaluator.evaluate(result.model, test_triples)
print(results)
```

---

#### **Step 6: Visualize Predictions**
Use **NetworkX** or **Plotly** to display:
- Predicted edges.
- Confidence scores for new links.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Visualize predicted graph
G = nx.DiGraph()
G.add_edges_from([
    ("Paper A", "Paper B"),
    ("Paper X", "Paper Y")  # Predicted link
])

nx.draw(G, with_labels=True)
plt.show()
```

---

### Tools to Explore
1. **PyTorch-Geometric**:
   - Lightweight, ideal for link prediction with GNNs.
   - Supports graph convolutional networks (GCNs) for advanced embeddings.

2. **PyKEEN**:
   - High-level library for knowledge graph embeddings.
   - Great for quick experiments with models like TransE or RotatE.

3. **DGL (Deep Graph Library)**:
   - Scalable graph representation and training.

---

# TUTORIAL PART 2 (AFTER DATA IS COLLECTED)
## Step-by-step to set up a pipeline in **PyKEEN** 
Let's start by outlining the steps, and we can dive into the details for each-

### **Step 1: Install Required Libraries**
First, ensure you have the necessary tools installed:
- **For PyKEEN:**
  ```bash
  pip install pykeen
  ```
- **For PyTorch-Geometric:**
  ```bash
  pip install torch torch-geometric
  ```
  Additional dependencies for PyTorch-Geometric vary depending on your system and GPU configuration. Follow the [installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### **Step 2: Load and Prepare Data**
We’ll organize the dataset into triples (`subject`, `predicate`, `object`) and split it into training and testing sets. PyKEEN and PyTorch-Geometric handle data differently:
- **PyKEEN** expects triples as a list or dataframe.
- **PyTorch-Geometric** requires edge indices and features in tensor format.

### **Step 3: Define the Model**
- **In PyKEEN**:
  PyKEEN offers pre-built models like TransE, RotatE, and DistMult. We can specify the model and its hyperparameters in a few lines of code.
- NEXT: **In PyTorch-Geometric**:
  You’d need to implement the graph neural network (GNN) or use prebuilt layers like GCN or GraphSAGE.

Let's start with a prebuilt model (e.g., TransE in PyKEEN)

### **Step 4: Train the Model**
- Train the model on the knowledge graph triples.
- Generate embeddings for entities and relationships.

### **Step 5: Evaluate the Model**
We’ll compute evaluation metrics like Mean Rank (MR), Mean Reciprocal Rank (MRR), and Hits@K to assess link prediction performance. PyKEEN has built-in evaluation functions, while PyTorch-Geometric requires custom metric implementation.

### **Step 6: Visualize and Analyze**
We’ll visualize:
1. The graph structure and clusters.
2. Embedding spaces to interpret relationships.
3. Predicted links and their significance.

---

# LET'S GET STARTED

Let’s dive into configuring the **PyKEEN pipeline** with an example dataset for your link prediction task.

---

### **Step 1: Setup Example Data**
We’ll create a toy dataset mimicking your academic citation network. The data will consist of triples like:
- `("Paper_A", "cites", "Paper_B")`
- `("Paper_A", "authored_by", "Author_X")`

```python
from pykeen.datasets import Dataset

# Define your triples as a list of tuples
triples = [
    ("Paper_A", "cites", "Paper_B"),
    ("Paper_B", "cites", "Paper_C"),
    ("Paper_A", "authored_by", "Author_X"),
    ("Paper_C", "authored_by", "Author_Y"),
]

# Save triples to a TSV file (required by PyKEEN for custom datasets)
import pandas as pd
df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
df.to_csv("citation_triples.tsv", sep="\t", index=False, header=False)

# Load as a PyKEEN Dataset
from pykeen.datasets import SingleTabbedDataset
dataset = SingleTabbedDataset("citation_triples.tsv")
```

---

### **Step 2: Configure the PyKEEN Pipeline**
The pipeline automates data splitting, model training, and evaluation. We’ll use **TransE** as the embedding model.

```python
from pykeen.pipeline import pipeline

# Run the pipeline
result = pipeline(
    model='TransE',  # Model for embeddings
    dataset=dataset,  # Use the custom dataset
    training_loop='slcwa',  # Standard negative sampling
    optimizer='Adam',  # Optimizer for gradient descent
    optimizer_kwargs=dict(lr=0.01),  # Learning rate
    loss='MarginRankingLoss',  # Loss function
    loss_kwargs=dict(margin=1.0),  # Margin for loss
    training_kwargs=dict(num_epochs=100, batch_size=32),  # Training params
    evaluation_kwargs=dict(batch_size=64),  # Evaluation params
    negative_sampler='basic',  # Negative sampling strategy
    random_seed=42,  # Reproducibility
)
```

---

### **Step 3: Analyze Training Results**
Once the pipeline completes, you can inspect the results and access the trained embeddings:

```python
# Print summary of results
print(result)

# Access the model
model = result.model

# Get entity and relation embeddings
entity_embeddings = model.entity_representations[0](indices=None)
relation_embeddings = model.relation_representations[0](indices=None)

print("Entity Embeddings Shape:", entity_embeddings.shape)
print("Relation Embeddings Shape:", relation_embeddings.shape)
```

---

### **Step 4: Visualize Embeddings**
Visualize the learned embeddings to interpret clusters or relationships:

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce embedding dimensions using PCA
pca = PCA(n_components=2)
entity_embeddings_2d = pca.fit_transform(entity_embeddings.detach().numpy())

# Scatter plot of embeddings
plt.figure(figsize=(8, 6))
plt.scatter(entity_embeddings_2d[:, 0], entity_embeddings_2d[:, 1], alpha=0.7)
plt.title("2D Visualization of Entity Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
```

---

### **Next Steps**
1. **Evaluation**: Analyze the pipeline’s built-in evaluation metrics (e.g., Hits@K, Mean Rank, Mean Reciprocal Rank).
2. **Predictions**: Use the trained model to predict missing links.
3. **Extend the Dataset**: Replace the toy dataset with a more realistic subset of your citation network.

Would you like to proceed with **model evaluation** or explore how to use the model for **predicting new links**? 

---

# MODEL EVALUATION POST
When evaluating your model, consider the following dimensions to ensure a thorough analysis of performance and usefulness:

---

### **1. Evaluation Metrics**
These are standard metrics for link prediction models:

#### **Mean Rank (MR)**
- Measures the average rank of the correct links in the model's predictions.
- Lower values indicate better performance.
- Example: If the true link ranks 1st, 5th, and 10th across three queries, the MR is \((1 + 5 + 10)/3 = 5.33\).

#### **Mean Reciprocal Rank (MRR)**
- Rewards high-ranking predictions by taking the reciprocal of the rank.
- Higher values indicate better performance.
- Formula:  
  \[
  MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}
  \]
- Example: If ranks are 1, 5, and 10, MRR is \((1 + 0.2 + 0.1)/3 = 0.433\).

#### **Hits@K**
- Fraction of correct links appearing in the top \(K\) predictions.
- Example: Hits@10 = 80% means 80% of true links are in the top 10 predictions.

#### **Optional: Precision, Recall, F1 Score**
- Particularly useful if you frame the task as a binary classification problem (link or no link).

---

### **2. Evaluate Negative Sampling**
Link prediction models require negative samples (non-existent links) for training and evaluation. Consider:
- Are your negative samples realistic? Example: Sampling disconnected but plausible nodes within the same domain.
- Are you balancing positive and negative samples in the evaluation?

---

### **3. Qualitative Evaluation**
#### **Predicted Links**
- Inspect a few predicted links to see if they make sense.
- Example: Does the model predict a citation between two highly related papers?

#### **Embedding Interpretability**
- Visualize the embedding space using techniques like PCA or t-SNE.
- Look for meaningful clustering of related nodes (e.g., papers grouped by subfield).

---

### **4. Baseline Comparison**
- Compare your model against simple baselines:
  - **Random Prediction**: Predict links randomly.
  - **Common Neighbors**: Rank pairs of nodes by the number of shared neighbors.
  - **Preferential Attachment**: Rank by product of node degrees.

---

### **5. Error Analysis**
- Analyze where the model struggles:
  - Are certain relation types harder to predict?
  - Do predictions deteriorate for long-tail (less frequent) nodes or relations?

---

### **6. Evaluation Strategy in PyKEEN**
PyKEEN simplifies evaluation by integrating the metrics directly into the pipeline. Here’s how to proceed:

```python
# Evaluate the model on the test set
from pykeen.evaluation import RankBasedEvaluator

evaluator = RankBasedEvaluator()
results = evaluator.evaluate(
    model=result.model,  # Trained model
    mapped_triples=dataset.testing.mapped_triples,  # Test triples
)

# Print evaluation results
print(results)
```

The `RankBasedEvaluator` outputs:
- Mean Rank (MR)
- Mean Reciprocal Rank (MRR)
- Hits@K (e.g., Hits@1, Hits@3, Hits@10)

---

### **7. Visualizing Evaluation Results**
Create a bar chart to compare Hits@K metrics:
```python
import matplotlib.pyplot as plt

# Example evaluation metrics
hits_at_k = {'Hits@1': 0.35, 'Hits@3': 0.55, 'Hits@10': 0.80}

# Bar chart
plt.bar(hits_at_k.keys(), hits_at_k.values(), alpha=0.7)
plt.title('Model Evaluation: Hits@K')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.show()
```

---

### **Next Steps**
- Run the evaluation in PyKEEN and inspect the results.
- If you’d like, we can focus on:
  1. **Error analysis** for the model.
  2. Creating **custom visualizations** of evaluation metrics.
  3. Comparing the model to baselines.


---

# GETTING STARTED WITH LINK PREDICTION
To predict new links using your trained model, we focus on using the model’s learned embeddings to evaluate the likelihood of a new or missing connection between nodes in the graph. Here’s how to approach it:

---

### **1. Use the Model for Scoring Potential Links**
Link prediction involves scoring pairs of nodes (triples) to determine the likelihood of a relationship. For instance:
- Predict if `("Paper_X", "cites", "Paper_Y")` is likely.

#### **A. Single Triple Prediction**
PyKEEN allows you to evaluate individual triples:
```python
# Example: Predict if Paper_A cites Paper_B
from pykeen.models import predict

triple = ('Paper_A', 'cites', 'Paper_B')
score = predict.predict_triples(triple, model=result.model)

print(f"Prediction score for {triple}: {score}")
```

#### **B. Batch Prediction**
Predict multiple links (e.g., across an entire test set or between specific nodes):
```python
# Batch predictions for a set of candidate triples
candidate_triples = [
    ('Paper_A', 'cites', 'Paper_B'),
    ('Paper_C', 'cites', 'Paper_D'),
    ('Paper_E', 'cites', 'Paper_F'),
]

scores = predict.predict_triples(candidate_triples, model=result.model)
for triple, score in zip(candidate_triples, scores):
    print(f"Triple: {triple}, Score: {score}")
```

Higher scores indicate stronger likelihoods of links.

---

### **2. Generate Candidate Links**
If you want to discover new links, generate plausible candidate triples:
- **Node Pair Sampling**: Consider pairs of nodes not connected in the graph.
- **Heuristic Sampling**: Use domain knowledge (e.g., papers in the same subfield) to generate candidates.

```python
import itertools

# Generate candidate triples between unconnected nodes
nodes = ['Paper_A', 'Paper_B', 'Paper_C', 'Paper_D']
relation = 'cites'

candidate_triples = [
    (node1, relation, node2)
    for node1, node2 in itertools.combinations(nodes, 2)
    if node1 != node2
]

print(f"Generated {len(candidate_triples)} candidate triples.")
```

---

### **3. Rank Predicted Links**
Once scores are calculated, rank the predicted links:
- Sort by score in descending order.
- Retain top-k links for further evaluation or validation.

```python
# Rank candidate triples by prediction score
ranked_triples = sorted(zip(candidate_triples, scores), key=lambda x: x[1], reverse=True)

print("Top predicted links:")
for triple, score in ranked_triples[:10]:
    print(f"Triple: {triple}, Score: {score}")
```

---

### **4. Validate Predicted Links**
To validate predictions:
- **Compare with Ground Truth**: Check if predicted links already exist but were withheld from training.
- **Expert Validation**: For new links (e.g., potential citations), rely on expert feedback or additional research.

---

### **5. Visualize Predicted Links**
Enhance interpretability by visualizing predicted links:
```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a NetworkX graph
G = nx.DiGraph()

# Add nodes and predicted edges
for triple, score in ranked_triples[:10]:
    head, relation, tail = triple
    G.add_edge(head, tail, label=relation, weight=score)

# Draw graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1500, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)})
plt.title("Predicted Links in Citation Network")
plt.show()
```

---

### **6. Next Steps**
- Validate the predictions through real-world testing or comparison with additional data.
- Use predicted links to uncover insights:
  - Missing citations that should connect papers.
  - Emerging clusters or new hubs in the citation network.

Would you like to implement a full workflow for **ranking and visualizing predictions**, or focus on a specific aspect like **candidate generation** or **validation**? Let me know!
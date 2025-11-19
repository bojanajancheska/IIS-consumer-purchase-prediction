# H&M Co-Purchase Prediction

This project uses graph neural networks to predict which H&M articles are likely to be purchased together. The system analyzes historical transaction data to identify co-purchase patterns and learns embeddings that capture the relationships between products. The model achieves a test AUC of 0.87, demonstrating strong predictive performance for item co-occurrence.

## Dataset

The project uses the [RelBench H&M dataset](https://relbench.stanford.edu/), which contains:

- Customer transactions
- Article metadata (product type, color, department, etc.)
- Over 15 million transactions across 58,929 unique articles

## Graph Representation

The data is structured as a weighted undirected graph where:

**Nodes:** Each node represents a unique H&M article (58,929 nodes total)

**Node Features:** Each article is represented by 9 encoded features from the article metadata:

- Product name
- Product type number
- Graphical appearance number
- Color group code
- Index code
- Index group number
- Section number
- Garment group number

**Edges:** An edge exists between two articles if they have been co-purchased (bought together in the same transaction) at least 2 times

**Edge Weights:** The weight of each edge represents the number of times those two items have been purchased together. For example, if articles A and B were bought together 15 times, the edge weight is 15. This captures the strength of the co-purchase relationship.

The graph contains approximately 7 million co-purchase pairs across the training, validation, and test sets.

## Setup

### Local Setup

1. Install [uv](https://github.com/astral-sh/uv) package manager (if not already installed)

2. Initialize the project and install dependencies:

```bash
uv sync
```

### Google Colab

The notebooks can be run independently on Google Colab. Each notebook includes setup cells that install the required dependencies.

## Usage

### Running Locally

**Important:** Run the notebooks in the following order:

1. **First, run `data_preprocessing.ipynb`:**

   - Downloads the RelBench H&M dataset
   - Processes transactions into co-purchase pairs
   - Creates graph structure with article features
   - Saves processed data to `output/data.pt`

2. **Then, run `model.ipynb`:**
   - Loads preprocessed data from `output/data.pt`
   - Trains a GraphSAGE-based GNN model
   - Evaluates model performance
   - Saves the best model to `output/best_model.pt`

### Running on Google Colab

Upload either notebook to Google Colab and run all cells. Each notebook is self-contained and will handle dependency installation automatically.

## Model Architecture

The model uses a lightweight Graph Neural Network with:

- 2-layer GraphSAGE convolution
- Batch normalization and dropout for regularization
- Link prediction via dot product of learned node embeddings
- Binary cross-entropy loss for training

## Results

- **Validation AUC:** 0.89
- **Test AUC:** 0.87
- **Test AP:** 0.85

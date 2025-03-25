# Hybrid Search with Reranking

This repository demonstrates a powerful hybrid search implementation using Qdrant vector database and multiple embedding models to search through a dataset of more than 1 million news articles. The system combines dense embeddings, sparse embeddings (BM25), and late interaction models to achieve high-quality search results with reranking capabilities.

## Features

- **Hybrid Search Architecture**: Combines multiple search approaches for optimal results
  - Dense embeddings (all-MiniLM-L6-v2)
  - Sparse embeddings (BM25)
  - Late interaction model (ColBERT v2.0)
- **Efficient Vector Storage**: Uses Qdrant for scalable vector search
- **Reranking**: Implements ColBERT v2.0 for high-quality result reranking
- **Filtering**: Supports filtering results by user ID
- **Scalability**: Includes resharding functionality for horizontal scaling
- **Performance Optimizations**: Binary quantization for reduced memory usage

## Architecture

The system uses a three-stage search approach:

1. **Initial Retrieval**: Uses both dense embeddings (all-MiniLM-L6-v2) and sparse embeddings (BM25) to retrieve candidate documents
2. **Reranking**: Applies ColBERT v2.0 late interaction model to rerank the candidates
3. **Filtering**: Optional filtering based on metadata (e.g., user ID)

![Hybrid Search Architecture]([https://i.imgur.com/placeholder.png](https://qdrant.tech/documentation/examples/reranking-hybrid-search/image3.png))

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/hybrid-search-reranking.git
cd hybrid-search-reranking
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install and run Qdrant:

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant
```

4. Download the dataset:

```bash
# Download the ABC News dataset from Kaggle:
# https://www.kaggle.com/datasets/therohk/million-headlines
# Place it in dataset/abcnews-date-text.csv
```

## Usage

### Data Ingestion

To load the dataset and create embeddings:

```python

# This will load the dataset, generate embeddings, and upload to Qdrant
upload_data()
```

### Searching

Basic search:

```python


# Search for articles about "Italy world cup"
search("Italy world cup")
```

Search with user filtering:

```python
# Search for articles about "Italy world cup" for user ID 6
search("Italy world cup", filter=6)
```

### Resharding for Performance

To create a sharded collection for better performance:

```python
from hybrid-search import resharding

# Create a sharded collection with 3 shards and replication factor 2
resharding()
```

## How It Works

### Embedding Models

The system uses three different embedding models:

1. **Dense Embedding**: `sentence-transformers/all-MiniLM-L6-v2`

   - General-purpose text embeddings
   - Good for semantic similarity

2. **Sparse Embedding**: `Qdrant/bm25`

   - Keyword-based sparse vectors
   - Excellent for term-matching

3. **Late Interaction**: `colbert-ir/colbertv2.0`
   - Token-level interactions between query and document
   - Superior reranking capabilities

### Search Process

1. The query is embedded using all three models
2. Initial candidates are retrieved using both dense and sparse vectors (prefetch)
3. The late interaction model reranks the candidates
4. Optional filtering is applied
5. Top results are returned

## Performance Considerations

- **Binary Quantization**: Reduces memory usage while maintaining search quality
- **Sharding**: Horizontal scaling for large datasets
- **Replication**: Improves availability and read throughput
- **Batch Processing**: Data is ingested in batches of 100 documents


## License

[MIT License](LICENSE)

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for the vector database
- [FastEmbed](https://github.com/qdrant/fastembed) for the embedding models
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) for the late interaction model

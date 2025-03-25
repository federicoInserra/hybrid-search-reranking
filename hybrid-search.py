from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client.models import Distance, VectorParams, models, PointStruct
from qdrant_client import QdrantClient
import csv
import random


def load_dataset():

    documents = []
    id = 0
    with open('dataset/abcnews-date-text.csv', mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            id += 1

            doc = {
                "publish_date": line[0],
                "headline_text": line[1],
                "doc_id": id
            }

            documents.append(doc)

    print("example of document")
    print(documents[1])

    return documents


def generate_embeddings(docs):

    dense_embeddings = list(dense_embedding_model.embed(
        doc["headline_text"] for doc in docs))

    bm25_embeddings = list(bm25_embedding_model.embed(
        doc["headline_text"] for doc in docs))

    late_interaction_embeddings = list(
        late_interaction_embedding_model.embed(doc["headline_text"] for doc in docs))

    return dense_embeddings, bm25_embeddings, late_interaction_embeddings


def create_collection(dense, bm25, late):

    client.create_collection(
        "hybrid-search",
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=len(dense[0]),
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=len(late[0][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF
                                              )
        },
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True,
            )),
    )


def upload_data():

    # Load dataset
    documents = load_dataset()

    # Start ingestion in batch
    dense_embeddings = []
    bm25_embeddings = []
    late_interaction_embeddings = []

    for i in range(100, len(documents), 100):

        print(f"Documents inserted {i}")
        batch = documents[i-100:i]

        print(batch[-1])

        dense_embeddings, bm25_embeddings, late_interaction_embeddings = generate_embeddings(
            batch)

        if not client.collection_exists("hybrid-search"):
            create_collection(dense_embeddings, bm25_embeddings,
                              late_interaction_embeddings)

        points = []
        for j in range(len(batch)):

            doc = batch[j]

            point = PointStruct(
                id=doc["doc_id"],
                vector={
                    "all-MiniLM-L6-v2": dense_embeddings[j],
                    "bm25": bm25_embeddings[j].as_object(),
                    "colbertv2.0": late_interaction_embeddings[j],
                },
                payload={"headline": doc["headline_text"],
                         "publish_date": doc["publish_date"],
                         "user_id": random.randint(1, 10)}
            )
            points.append(point)

        operation_info = client.upsert(
            collection_name="hybrid-search",
            points=points
        )


def search(query, filter=False):

    dense_vectors = next(dense_embedding_model.query_embed(query))
    sparse_vectors = next(bm25_embedding_model.query_embed(query))
    late_vectors = next(late_interaction_embedding_model.query_embed(query))

    prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="bm25",
            limit=20,
        ),
    ]

    query_filter = {}
    if filter != False:

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(
                        value=filter,
                    ),
                )
            ]
        )

    # Running the query
    results = client.query_points(
        "hybrid-search",
        prefetch=prefetch,
        query=late_vectors,
        using="colbertv2.0",
        query_filter=query_filter,
        with_payload=True,
        limit=10,
    )

    print(results)


def shard_collection():

    client.create_collection(
        "hybrid-search-sharded",
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF
                                              )
        },
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True,
            )),
        init_from=models.InitFrom(collection="hybrid-search"),
        shard_number=3,
        replication_factor=2
    )


# prepare client
client = QdrantClient(url="http://localhost:6333")


# Load models
dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding(
    "colbert-ir/colbertv2.0")


# Loading dataset and creating the embedding. This could take a while....
upload_data()

# Running the search
query = "Italy world cup"
# filter_user = 6

search(query, filter_user=False)

# If you want you can also deploy the collection on multiple nodes and shards
# shard_collection()

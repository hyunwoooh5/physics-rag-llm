import uuid
import pandas as pd

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from tqdm.auto import tqdm

COLLECTION_NAME = "physics_rag_collection_lat_nuc"
VECTOR_SIZE = 768

TARGET_PRIMARY_CATEGORIES = ["nucl-th", "hep-lat"]

# Start the client
client = QdrantClient("http://localhost:6333")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "default": models.VectorParams(  # vector_name = 'default'
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    }
)
print(f"Collection '{COLLECTION_NAME}' created!")

embedding_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5", threads=4)


def get_vector(text):
    return next(embedding_model.embed([text])).tolist()


def generate_uuid_from_string(s):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(s)))


def get_primary_category(categories_str):
    if not categories_str:
        return None

    return categories_str.split(' ')[0]


def ingest_data(reader):

    for df in tqdm(reader):
        df['primary_category'] = df['categories'].apply(get_primary_category)

        filtered_df = df[df['primary_category'].isin(
            TARGET_PRIMARY_CATEGORIES)]

        points_batch = []

        for _, row in filtered_df.iterrows():
            arxiv_id = row['id']
            title = row['title']
            abstract = row['abstract']
            preprint_date = row['update_date']
            authors = row['authors_parsed']
            primary_category = row['primary_category']

            point_id = generate_uuid_from_string(arxiv_id)

            vector = get_vector(abstract)

            payload = {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "preprint_date": preprint_date,
                "authors": authors,
                "primary_category": primary_category
            }

            point = models.PointStruct(
                id=point_id,
                vector={"default": vector},
                payload=payload
            )
            points_batch.append(point)

        if points_batch:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch
            )

    print("Ingestion Finished!")


if __name__ == "__main__":
    reader = pd.read_json(
        "data/arxiv-metadata-oai-snapshot.json", lines=True, chunksize=10000, dtype={'id', str})
    ingest_data(reader)

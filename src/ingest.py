import time
import requests
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from tqdm.auto import tqdm

COLLECTION_NAME = "physics_rag_collection_lat_nuc"
VECTOR_SIZE = 768  # dimension of the latent space
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


def ingest_data(target_count=10):
    base_url = "https://inspirehep.net/api/literature"
    page = 1
    count = 0

    pbar = tqdm(total=target_count)

    while count < target_count:
        params = {
            "q": "(primary_arxiv_category:nucl-th OR primary_arxiv_category:hep-lat)",
            "size": 500,
            "page": page,
            "sort": "mostrecent"
        }

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 400:  # usually up to 10,000
                break
            response.raise_for_status()
            data = response.json()
            hits = data.get("hits", {}).get("hits", [])

            if not hits:
                break

            points_batch = []

            for hit in hits:
                metadata = hit.get("metadata", {})

                abstract = ""
                if "abstracts" in metadata and len(metadata["abstracts"]) > 0:
                    abstract = metadata["abstracts"][0].get("value", "")

                title = metadata.get("titles", [{}])[0].get("title", "")

                preprint_date = metadata.get("preprint_date")

                if not abstract:
                    continue

                vector = get_vector(abstract)

                point = models.PointStruct(
                    id=int(hit.get("id")),
                    vector={"default": vector},
                    payload={
                        "title": title,
                        "abstract": abstract,
                        "preprint_date": preprint_date,
                        "authors": len(metadata.get("authors", []))
                    }
                )
                points_batch.append(point)

            # batch upload
            if points_batch:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_batch
                )
                count += len(points_batch)
                pbar.update(len(points_batch))

            page += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"Error at page {page}: {e}")
            break

    pbar.close()
    print("Ingestion Finished!")


if __name__ == "__main__":
    ingest_data(target_count=10000)

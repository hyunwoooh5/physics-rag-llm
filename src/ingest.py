import time
import requests
import uuid
import urllib
import feedparser

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from tqdm.auto import tqdm

COLLECTION_NAME = "physics_rag_collection_lat_nuc"
VECTOR_SIZE = 768  # dimension of the latent space

BATCH_SIZE = 1000

TARGET_PRIMARY_CATEGORIES = {"nucl-th", "hep-lat"}

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
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def ingest_data(target_count=1000):
    base_url = "https://export.arxiv.org/api/query?"

    search_query = "cat:nucl-th OR cat:hep-lat"

    pbar = tqdm(total=target_count)

    current_count = 0
    start_index = 0

    while current_count < target_count:
        params = (
            f"search_query={urllib.parse.quote(search_query)}&"
            f"start={start_index}&"
            f"max_results={BATCH_SIZE}"
            f"&sortBy=submittedDate&sortOrder=descending"

        )

        url = base_url + params

        try:
            feed = feedparser.parse(url)

            if not feed.entries:
                print("No more entries found")
                break

            points_batch = []

            for entry in feed.entries:
                try:
                    primary_cat = entry.arxiv_primary_category['term']
                except AttributeError:
                    if 'tags' in entry and len(entry.tags) > 0:
                        primary_cat = entry.tags[0]['term']
                    else:
                        continue

                if primary_cat not in TARGET_PRIMARY_CATEGORIES:
                    continue

                title = entry.title.replace('\n', ' ')
                abstract = entry.summary.replace('\n', ' ')
                preprint_date = entry.published
                authors = [author.name for author in entry.authors]

                arxiv_id = entry.id.split('/abs/')[-1]
                point_id = generate_uuid_from_string(arxiv_id)

                vector = get_vector(abstract)

                payload = {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "preprint_date": preprint_date,
                    "authors": authors,
                    "link": entry.id

                }

                point = models.PointStruct(
                    id=point_id,
                    vector={"default": vector},
                    payload=payload
                )
                points_batch.append(point)

            raw_fetched_count = len(feed.entries)
            saved_count = len(points_batch)

            # batch upload
            if points_batch:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_batch
                )

                
                current_count += saved_count
                pbar.update(saved_count)
            
            start_index += raw_fetched_count

            time.sleep(3.0)

        except Exception as e:
            print(f" Error at start_index {start_index}: {e}")

            time.sleep(10)
            continue

    pbar.close()
    print("Ingestion Finished!")


if __name__ == "__main__":
    ingest_data(target_count=5000)

# Physics Research Assistant

**Retrieval Augmented Generation (RAG) for High-Energy & Nuclear Physics**

## Overview

This project implements a Retrieval Augmented Generation (RAG) system designed to mitigate information overload for physics researchers. By leveraging Large Language Models (LLMs) grounded in specific arXiv datasets, this tool allows researchers to query complex scientific concepts and receive answers based strictly on peer-reviewed or pre-print literature.

Unlike standard keyword search, this system utilizes dense vector embeddings to understand the semantic context of queries (e.g., distinguishing between "lattice" in condensed matter vs. QCD).


## Dataset

The dataset is populated using one of the following methods:

* **arXiv API**: For fetching the most recent papers directly from arXiv.
* **Kaggle arXiv Dataset**: For bulk ingestion of historical data using the [Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).

**Primary Categories:**
* `nucl-th` (Nuclear Theory)
* `hep-lat` (Lattice Field Theory)

* **Content:** Title, Abstract, and Preprint Date.

*Note: While the current deployment focuses on Nuclear and Lattice theory, the ingestion pipeline in [src/ingest.py](src/ingest.py) is modular. It can be adapted for other arXiv categories such as `hep-th` (High Energy Physics - Theory) or `gr-qc` (General Relativity and Quantum Cosmology).*


## Technologies

* **Runtime & Package Management:** `uv`
* **Vector Database:** `qdrant`
* **Embeddings:** `fastembed` (Dense retrieval)
* **LLM:** `Google Gemini`
* **API:** `FastAPI`
* **Frontend:** `Streamlit`
* **Observability:** `Phoenix` (Arize AI)


## Preparation


### 1. Installation

Ensure `uv` is installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

Install dependencies with a locked environment:

```bash
uv sync --locked
```

### 2. Environment Configuration

Launch Qdrant to save data in a vector store:

```bash
docker run --rm -p 6333:6333 -p 6334:6334 \
   -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
   qdrant/qdrant
```


### 3. Ingestion

You can populate the Qdrant vector store using either the API or the local JSON dataset.

**Option A: Ingest via arXiv API (Default)**
Best for fetching the latest papers or specific queries.

```bash
uv run python src/ingest.py
```



**Option B: Ingest via Kaggle Dataset**
Best for large-scale bulk ingestion.

1. Download `arxiv-metadata-oai-snapshot.json` from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
2. Place the JSON file in [data](data/).
3. Run the JSON ingestion script:

```bash
uv run python src/ingest_json.py
```




## Running the application (with Docker-Compose)

### 1. Environment Configuration

Create a `.env` file and append your Gemini API key:

```bash
echo '\nGEMINI_API_KEY=your_api_key_here' >> .env

```


### 2. Docker Deployment

Launch the services (API, Database, and Monitoring) in detached mode:

```bash
docker compose up -d

```


## Using the Application

### CLI (cURL)

You can query the RAG pipeline directly via the terminal. Note that the API is exposed on port 8000.

```bash
curl -X POST "http://localhost:8000/rag" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the applications of control variates in lattice QCD?"}'

```


### Web Interface

* **Chat Interface (Streamlit):** [http://localhost:8501](http://localhost:8501)
* **REST API Documentation (Swagger UI):** [http://localhost:8080/docs](http://localhost:8080/docs) (mapped via Docker)



## Experiments & Evaluation

We performed rigorous evaluation on both the retrieval component and the generation (RAG) flow.



### Retrieval Evaluation

We utilized two key metrics:

1. **Hit Rate:** The percentage of queries where the relevant document appeared in the retrieved results.
2. **MRR (Mean Reciprocal Rank):** A measure of how high the relevant document ranks in the list (1.0 is perfect, meaning the top result was correct).

**Dense Search Performance:**

| Top_k | Hit Rate | MRR |
| --- | --- | --- |
| 1 | 0.746239 | 0.746239 |
| 3 | 0.875627 | 0.804580 |
| **5** | **0.907723** | **0.811852** |
| 10 | 0.923771 | 0.814037 |
| 20 | 0.944835 | 0.815637 |

**Decision:** We selected `top_k=5`.

* **Reasoning:** Increasing `top_k` from 5 to 10 yields only a ~1.6% increase in Hit Rate but doubles the context tokens passed to the LLM, increasing latency and cost. `Top_k=5` offers the optimal trade-off between recall (90.7%) and context window efficiency.


**Reranking Analysis:**
We evaluated adding a Cross-Encoder reranker.

* *Performance:* Hit Rate: 0.898, MRR: 0.834 (at top_k=50 retrieval).
* *Decision:* **Rejected.** While MRR improved, the latency increased by >10x. For this iteration, we prioritize response time over the marginal gain in ranking precision.




### RAG Flow Evaluation (LLM-as-a-Judge)

We employed an LLM-as-a-Judge approach to score the generated answers on **Relevance** (does it answer the query?) and **Faithfulness** (is it grounded in the retrieved context?).

| Model | Relevance | Faithfulness |
| --- | --- | --- |
| **Gemini-2.5-Flash-Lite** | 0.805 | 0.969 |
| Gemini-2.5-Flash | 0.820 | 0.980 |

**Decision:** We selected **Gemini-2.5-Flash-Lite**.

* **Reasoning:** Although the standard Flash model performs slightly better (+1.5% relevance, +1.1% faithfulness), the Lite model is significantly faster (~6x throughput) and more cost-effective. The faithfulness score of 0.969 is deemed sufficient for a research assistant context.




## Monitoring

We use **Phoenix** for tracing execution, debugging retrieval context, and monitoring LLM latency.

* Dashboard: [http://localhost:6006](http://localhost:6006)




## Project Structure

```plaintext
├── data
│   ├── ground_dataset_test.csv           # Initial test dataset
│   ├── ground_truth_dataset.csv          # Ground truth for evaluation
│   ├── rag_results_eval_flash-lite.json  # Eval outputs (Flash-Lite)
│   ├── rag_results_eval_flash.json       # Eval outputs (Flash)
│   └── rag_results_test.json             # Test dataset outputs
├── docker-compose.yaml                   # Container orchestration
├── Dockerfile                            # Container for deployment
├── Dockerfile.ui                         # Container for deployment (Streamlit UI)
├── LICENSE
├── notebooks
│   ├── ground_truth.ipynb                # Ground truth generation logic
│   ├── rag.ipynb                         # RAG pipeline prototyping
│   └── test.ipynb                        # Testing playground
├── pyproject.toml                        # Project configuration & dependencies
├── README.md
├── src
│   ├── ingest_json.py                    # ETL pipeline (json -> Qdrant)
│   ├── ingest.py                         # ETL pipeline (arXiv -> Qdrant)
│   ├── rag.py                            # RAG inference logic
│   ├── ui.py                             # Streamlit frontend application
│   └── serve.py                          # FastAPI application
├── test.py                               # Unit/Integration tests
└── uv.lock                               # Dependency lockfile

```



## License

This project is licensed under the MIT License.

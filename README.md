# cs5416-ml-pipeline

## Development Environment

### Install uv

```bash
brew install uv
```

or

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Python Virtual Environment

```bash
uv venv
```

### Git Hooks

_run inside the repo_

```bash
brew install prek
prek install
```

or

```bash
uv tool install prek
prek install
```

You can run hooks with

```bash
prek run --all-files
```

## Running the Pipeline

Generate the documents

```bash
python ./scripts/create_test_docs.py
```

terminal 1

```bash
./start_pipeline.sh
```

terminal 2
_the client.py runs using the system install python and requires that the system installed python has requests installed_

```bash
./scripts/client.py
```

## Profiling

First install Docker or Podman (_Podman can replace any Docker command_). You might also need to install Docker compose or Podman compose as well.

Run the profiling service

```bash
docker compose -f monitoring/docker-compose.yml up
```

Access the profiling service frontend by going to http://localhost:3000

## Modular Profiling & Experiments

### Adding a Profile

Create a YAML file (e.g., `my_profile.yaml`) with roughly the following structure (look at experiments for more examples):

```yaml
name: my_custom_profile
description: "Custom profile description"
batch_size: 32
batch_timeout: 0.1
components:
  - name: orchestrator
    type: gateway
    config:
      retrieval_url: "http://localhost:8001"
routes:
  - target: gateway
    prefix: "/"
```

### Running Experiments

Use `scripts/run_experiment.py` to launch a full experiment with profiling:

```bash
python scripts/run_experiment.py configs/experiments/baseline.yaml
```

This will:

1. Load the experiment manifest.
2. Start the monitoring stack (if not running).
3. Start the cluster nodes with the specified profiles.
4. Run the workload (profile_pipeline.py).
5. Collect metrics and process stats.
6. Save artifacts to `artifacts/experiments/<run_id>_<timestamp>/`.

## Configuration

### Environment Variables

- `DOCUMENTS_PAYLOAD_MODE`: Controls how documents are passed between nodes.
  - `full` (default): Full document content is sent.
  - `id_only`: Only document IDs are sent. Requires a `DocumentStore` on the receiving node.
  - `compressed`: Documents are compressed before sending.

### Running All Experiments

To run all experiments defined in `configs/experiments/`:

```bash
./scripts/run_all_experiments.sh
```

### Analysis

To aggregate results and generate plots:

```bash
python scripts/analyze_experiments.py
```

Results will be in `analysis/`. Also, make sure to delete the old experiments or duplicate experiments to make things less confusing.

## Caching

The pipeline supports multi-level caching to improve performance.

### Configuration

Enable caching via environment variables or `.env` file:

- `GATEWAY_RESPONSE_CACHE_ENABLED`: Enable gateway response caching (default: False)
- `CACHE_MAX_TTL`: Max TTL for caches in seconds (default: 60.0)
- `DISABLE_CACHE_FOR_PROFILING`: Force disable caches during profiling (default: True)
- `GATEWAY_CACHE_CAPACITY`: Capacity of the gateway response cache (default: 1000)
- `FUZZY_CACHE_MATCHING`: Enable fuzzy matching (token sort) for gateway cache (default: False)

### Profiling with Cache

To profile with caching enabled:

1. Set `DISABLE_CACHE_FOR_PROFILING=False`.
2. Use `scripts/profile_pipeline.py` with `--randomize-queries` to bypass cache or `--clear-cache` to clear cache before run.

```bash
python scripts/profile_pipeline.py --randomize-queries
```

### Cache Levels

1. **Gateway Response Cache**: Caches full query responses.
2. **Embedding Cache**: Caches generated embeddings for text.
3. **Retrieval Result Cache**: Caches FAISS search results (doc IDs and scores) for embeddings.
4. **Document Fetch Cache**: Caches document content by ID.

### Metrics

Cache metrics are exposed via Prometheus:
- `pipeline_cache_hits_total`
- `pipeline_cache_misses_total`
- `pipeline_cache_evictions_total`

## Performance Tuning

The pipeline exposes several environment variables to tune CPU parallelism and concurrency:

- `CPU_INFERENCE_THREADS`: Number of threads for CPU inference (torch, faiss). Defaults to `min(16, cpu_count)`.
- `CPU_WORKER_THREADS`: Number of threads for worker pools (retrieval fetch, etc.). Defaults to `min(16, cpu_count)`.
- `MAX_PARALLEL_GENERATION`: Maximum number of concurrent generation requests. Defaults to 4.

Recommended settings:
- **Grader/Server**: `CPU_INFERENCE_THREADS=16`, `CPU_WORKER_THREADS=16`, `MAX_PARALLEL_GENERATION=4`
- **Laptop**: `CPU_INFERENCE_THREADS=6`, `CPU_WORKER_THREADS=6`, `MAX_PARALLEL_GENERATION=2`

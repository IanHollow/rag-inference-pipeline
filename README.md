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

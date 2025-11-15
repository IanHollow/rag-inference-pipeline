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
./run_pipeline.sh
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

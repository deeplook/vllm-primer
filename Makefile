.DEFAULT_GOAL := help

EXAMPLE ?= examples/01_check_endpoint.py
PORT ?= 8000
AWS_GPU := uv run python tools/aws_gpu.py
LESSONS := $(sort $(wildcard examples/[0-9]*.py))

.PHONY: help install run format lint typecheck test check-all mock hooks \
        live-plan live-up live-status live-wait live-logs live-forward live-run \
        clean-remote clean-remote-all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install locked dependencies
	uv sync --all-groups --locked

run: ## Run one lesson against $$VLLM_BASE_URL (EXAMPLE=examples/03_streaming.py)
	uv run python $(EXAMPLE)

format: ## Format and fix Python code
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Check formatting and linting
	uv run ruff format --check .
	uv run ruff check .

typecheck: ## Strict mypy over examples, tests and tools
	uv run mypy examples tests tools

test: ## Offline tests: lessons against the mock server, AWS tool against fakes
	uv run pytest

check-all: lint typecheck test ## Safe quality gate (offline, never modifies files)

mock: ## Start the fake vLLM server for offline lessons (PORT=8000)
	uv run python tools/mock_server.py $(PORT)

hooks: ## Install the git pre-commit hook (needs uv)
	uvx pre-commit install

live-plan: ## AWS: check access, quota and price; creates nothing
	$(AWS_GPU) plan

live-up: ## AWS: launch the GPU instance (starts billing; add YES=1 to confirm)
	$(AWS_GPU) up $(if $(YES),--yes)

live-status: ## AWS: list the instance, uptime and cost so far
	$(AWS_GPU) status

live-wait: ## AWS: wait until vLLM answers /health
	$(AWS_GPU) wait

live-logs: ## AWS: show the vLLM container logs
	$(AWS_GPU) logs

live-forward: ## AWS: tunnel localhost:8000 to the instance (keep running)
	$(AWS_GPU) forward

live-run: ## Run every lesson against $$VLLM_BASE_URL (real or mock server)
	@for ex in $(LESSONS); do echo "==> $$ex"; uv run python $$ex || exit 1; done

clean-remote: ## AWS: terminate one instance (INSTANCE=i-... VLLM_PRIMER_CONFIRM=terminate)
	$(AWS_GPU) down $(INSTANCE)

clean-remote-all: ## AWS: delete leftover security group, role, token (VLLM_PRIMER_CONFIRM=purge)
	$(AWS_GPU) purge

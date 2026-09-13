.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint fmt typecheck test cov check clean 	docker-build compose-up compose-down k8s-up k8s-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | 		awk 'BEGIN {FS = ":.*?## "} {printf "  [36m%-16s[0m %s
", $$1, $$2}'

install: ## Create the venv and install all extras
	$(UV) sync --all-extras

lint: ## Lint with ruff
	$(UV) run ruff check src tests

fmt: ## Format with ruff
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck: ## Type-check with mypy (strict)
	$(UV) run mypy

test: ## Run the test suite (no API key, no network)
	$(UV) run pytest

cov: ## Run tests with coverage gate
	$(UV) run pytest --cov --cov-report=term-missing --cov-report=xml

check: lint typecheck cov ## Everything CI runs

clean: ## Remove build and cache artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

docker-build: ## Build all images
	docker compose -f deploy/docker-compose.yml build

compose-up: ## Bring the full stack up locally
	docker compose -f deploy/docker-compose.yml up -d

compose-down: ## Tear the local stack down
	docker compose -f deploy/docker-compose.yml down -v

k8s-up: ## Deploy to the local minikube cluster
	kubectl apply -k deploy/k8s/overlays/local

k8s-down: ## Remove the local deployment
	kubectl delete -k deploy/k8s/overlays/local --ignore-not-found

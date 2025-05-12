
help: ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

pre-commit: ## Install pre commit hooks
	pre-commit install
	pre-commit install-hooks

format: ## Format with pre commit
	pre-commit run --all-files

bump: ## Update requirements
	uv sync
	uv pip compile pyproject.toml -o requirements.txt

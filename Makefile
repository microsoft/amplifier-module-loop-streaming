# Module Makefile
# Provides test target for integration with parent make test

.PHONY: test install clean

test: ## Run validation tests
	@uv run pytest tests/ -v

install: ## Install module in editable mode
	@uv pip install -e .

clean: ## Clean build artifacts
	@rm -rf __pycache__ *.egg-info .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {{}} + 2>/dev/null || true

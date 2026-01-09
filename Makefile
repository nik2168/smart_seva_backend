.PHONY: api test install help

# Default target
help:
	@echo "Available commands:"
	@echo "  make api      - Run the FastAPI server"
	@echo "  make test     - Run all tests"
	@echo "  make install  - Install dependencies"
	@echo "  make clean    - Clean cache files"

# Run the API server
api:
	@echo "Starting Smart Seva API server..."
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
test:
	@echo "Running all tests..."
	uv run python -m src.test_scripts.test_all

# Install dependencies
install:
	@echo "Installing dependencies..."
	uv sync

# Clean cache files
clean:
	@echo "Cleaning cache files..."
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cache cleaned!"
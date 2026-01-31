.PHONY: install test lint format clean docker-build

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8 mypy

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist build *.egg-info

docker-build:
	docker build -f docker/Dockerfile.api -t fraud-detection-api:latest .
	docker build -f docker/Dockerfile.training -t fraud-detection-training:latest .
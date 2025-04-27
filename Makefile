# Makefile for Drape Backend FastAPI Server

.PHONY: help setup run clean

help:
	@echo "Available targets:"
	@echo "  setup   - Create venv, install dependencies, and prompt for API key if needed"
	@echo "  run     - Start the FastAPI server (requires GOOGLE_API_KEY)"
	@echo "  clean   - Remove venv and __pycache__"

setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

run:
	./venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf venv __pycache__ *.pyc .pytest_cache .mypy_cache

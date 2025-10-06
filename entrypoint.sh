#!/bin/sh

uv run gunicorn food_access_model.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --timeout 600

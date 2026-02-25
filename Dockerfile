FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies (only required ones for backend, skip heavy ML packages if desirable for smaller image, but we'll use same for now)
RUN pip install --no-cache-dir fastapi uvicorn neo4j pydantic

COPY backend /app/backend

# Run on port 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

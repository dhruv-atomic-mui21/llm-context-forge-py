FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN pip install --upgrade pip
COPY pyproject.toml setup.py /app/
RUN pip install -e .

# Copy application files
COPY contextforge/ /app/contextforge/

EXPOSE 8000

# Start FastAPI server
CMD ["python", "-m", "contextforge.api.app"]

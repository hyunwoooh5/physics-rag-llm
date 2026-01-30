# Use the official lightweight Python 3.13 slim image (Debian Bookworm)
FROM python:3.13.11-slim-bookworm

# Copy the uv package manager from its latest image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory for the application
WORKDIR /code

# Add the virtual environment's bin directory to the PATH
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency and environment configuration files
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies exactly as pinned in uv.lock
RUN uv sync --locked --no-dev

# Copy application source files and model artifact
COPY "src/rag.py" "src/serve.py" ./

# Expose the application port
EXPOSE 8080

# Launch the app using Uvicorn (ASGI server)
ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_PATH="/opt/venv"

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory to /app
WORKDIR /app

# Copy dependency specification files
COPY pyproject.toml poetry.lock /app/

# Install dependencies (without installing the current package)
RUN poetry install --no-root --only main

# Copy the entire project into the container
COPY . /app/

# Set proper permissions for the virtual environment
RUN chmod -R 755 /opt/venv

# Add the virtual environment's bin directory to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user and adjust ownership for shared directories and the app folder
RUN useradd -m appuser \
    && mkdir /shared_data && chown -R appuser:appuser /shared_data \
    && chown -R appuser /app

# Switch to the non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Start the application using Uvicorn.
# Note: We use "app.main:app" because the main.py file is inside the app/ folder.
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

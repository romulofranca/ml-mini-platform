FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=false \
  POETRY_VIRTUALENVS_PATH="/opt/venv" \
  POETRY_VERSION=1.8.3

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN poetry install --no-root --only main

COPY . /app/

RUN chmod -R 755 /opt/venv

ENV PATH="/opt/venv/$(basename $(poetry env list --full-path | grep Activated | awk '{print $1}'))/bin:$PATH"

RUN useradd -m appuser

RUN mkdir /shared_data && chown -R appuser:appuser /shared_data

RUN chown -R appuser /app

USER appuser

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
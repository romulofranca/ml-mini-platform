import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routes import dataset, model
from app.middleware.error_handler import (
    http_exception_handler,
    generic_exception_handler,
)
from app.db import models, session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/ml_registry.db")

    if DATABASE_URL.startswith("sqlite"):
        db_file = DATABASE_URL.replace("sqlite:///", "")
        if not os.path.exists(db_file):
            models.Base.metadata.create_all(bind=session.engine)
            logger.info(
                "SQLite database file not found. Database and tables created."
            )
        else:
            logger.info("SQLite database file exists. Skipping creation.")
    else:
        models.Base.metadata.create_all(bind=session.engine)
        logger.info("Non-SQLite database: tables created.")
    yield


app = FastAPI(
    title="ML Mini Platform",
    description=(
        "A platform for cataloging datasets, training models, "
        "promoting them, and serving predictions"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset.router)
app.include_router(model.router)

app.add_exception_handler(Exception, generic_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)


@app.get("/", summary="Check API status")
def home():
    return {"message": "ML Mini Platform is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

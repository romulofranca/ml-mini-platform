import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.routes import dataset, model
from app.middleware.error_handler import (
    http_exception_handler,
    generic_exception_handler,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Mini Platform",
    description="A platform for training and serving ML models",
    version="0.1.0",
)

# CORS configuration – in production, restrict allowed origins as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(dataset.router)
app.include_router(model.router)

# Setup exception handlers
app.add_exception_handler(Exception, generic_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)


@app.get("/", summary="Check API status", tags=["General"])
def home():
    return {"message": "ML Mini Platform is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error occurred.")
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error"}
    )

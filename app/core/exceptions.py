from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base exception for expected application errors."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class UnsupportedFileTypeError(AppError):
    """Raised when an unsupported file type is uploaded."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class FileTooLargeError(AppError):
    """Raised when an uploaded file exceeds the size limit."""

    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class EmptyDocumentError(AppError):
    """Raised when no usable text can be extracted."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for application and unexpected errors."""

    @app.exception_handler(AppError)
    async def handle_app_error(
        request: Request,
        exc: AppError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "detail": "Something went wrong.",
            },
        )
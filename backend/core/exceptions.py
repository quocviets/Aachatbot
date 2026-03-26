"""
Custom exception hierarchy for the backend.
All exceptions here map to HTTP responses in the global exception handler.
"""


class AppException(Exception):
    """Base class for all application exceptions."""
    status_code: int = 500
    message: str = "Internal server error"

    def __init__(self, message: str | None = None):
        self.message = message or self.__class__.message
        super().__init__(self.message)


class FileTooLargeError(AppException):
    status_code = 413
    message = "File exceeds maximum allowed size"


class InvalidFileTypeError(AppException):
    status_code = 400
    message = "File type not supported. Use JPG, JPEG, or PNG"


class PlantNotSupportedError(AppException):
    status_code = 422
    message = "Plant type not supported"


class ModelNotReadyError(AppException):
    status_code = 503
    message = "AI model is not ready yet"


class InferenceError(AppException):
    status_code = 500
    message = "Inference failed unexpectedly"


class ImageNotFoundError(AppException):
    status_code = 404
    message = "Requested image not found"

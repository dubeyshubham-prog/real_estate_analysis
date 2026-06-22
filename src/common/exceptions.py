class EstateAIError(Exception):
    """Base exception for expected EstateAI application errors."""


class DataLoadError(EstateAIError):
    """Raised when a required dataset cannot be loaded safely."""


class DataValidationError(EstateAIError):
    """Raised when input data does not satisfy a pipeline requirement."""


class AnalysisError(EstateAIError):
    """Raised when an analysis report or visualization cannot be created."""


class UploadValidationError(EstateAIError):
    """Raised when an uploaded file is unsafe or unsupported."""


class ResourceNotFoundError(EstateAIError):
    """Raised when a requested application resource does not exist."""


class FeatureUnavailableError(EstateAIError):
    """Raised when an optional product feature is disabled."""

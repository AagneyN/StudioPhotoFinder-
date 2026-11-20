"""Domain exceptions so callers (API, CLI, UI) can report precise, user-friendly messages."""


class FinderError(Exception):
    """Base class."""


class ModelUnavailableError(FinderError):
    pass


class InvalidImageError(FinderError):
    pass


class NoFaceError(FinderError):
    pass


class IndexUnavailableError(FinderError):
    """FAISS index missing or incompatible with the current model/database."""


class DatasetNotFoundError(FinderError):
    pass

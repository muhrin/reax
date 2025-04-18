class MisconfigurationException(BaseException):
    """Exception raised when there is a problem with the configuration."""


class DataNotFound(BaseException):
    """Raised when data that was expected is not found, e.g. a key is missing from a dictionary
    or a value is unexpectedly `None`."""

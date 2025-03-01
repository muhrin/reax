from typing import Final

# Keys for the different categories of metrics
LOG: Final[str] = "log"
PBAR: Final[str] = "pbar"
LISTENER: Final[str] = "listener"


# Use to indicate that there is no upper maximum (e.g. max_batches=NO_LIMIT)
NO_LIMIT: Final[None] = None

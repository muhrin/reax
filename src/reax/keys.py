from typing import Final

# Keys for the different categories of metrics
LOG: Final[str] = "log"
PBAR: Final[str] = "pbar"
CALLBACK: Final[str] = "callback"


# Use to indicate that there is no upper maximum (e.g. max_batches=NO_LIMIT)
NO_LIMIT: Final[float] = float("inf")

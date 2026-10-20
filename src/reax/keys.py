from typing import Final, Literal

# Keys for the different categories of metrics
LOG: Literal["log"] = "log"
PBAR: Literal["pbar"] = "pbar"
LISTENER: Literal["listener"] = "listener"


# Use to indicate that there is no upper maximum (e.g. max_batches=NO_LIMIT)
NO_LIMIT: Final[None] = None

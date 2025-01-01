import pathlib
from typing import Union

import jax.typing
import jaxtyping as jt

Path = Union[str, bytes, pathlib.Path]
ArrayMask = Union[jt.Int[jt.ArrayLike, "..."], jt.Bool[jt.ArrayLike, "..."]]
MetricsDict = dict[str, jax.typing.ArrayLike]

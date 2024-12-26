import pathlib
from typing import Union

import jaxtyping as jt

Path = Union[str, bytes, pathlib.Path]
ArrayMask = Union[jt.Int[jt.ArrayLike, "..."], jt.Bool[jt.ArrayLike, "..."]]

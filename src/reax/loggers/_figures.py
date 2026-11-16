# Copyright (C) 2024  Martin Uhrin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Conversion of figures and images from common libraries into encoded image bytes.

None of the plotting libraries are imported unless an object from that library is actually passed
in, so none of them are hard dependencies of REAX.
"""

from collections.abc import Callable
import dataclasses
import io
from typing import Any, Final

import numpy as np

__all__ = ("FigureData", "register_figure_converter", "to_figure_data")

PNG_MAGIC: Final[bytes] = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC: Final[bytes] = b"\xff\xd8\xff"


@dataclasses.dataclass(frozen=True)
class FigureData:
    """An encoded image."""

    data: bytes
    fmt: str  # File extension without the dot, e.g. "png" or "jpeg"

    def to_array(self) -> np.ndarray:
        """Decode into an ``(H, W, C)`` ``uint8`` array (requires ``pillow``)."""
        pil_image = _import_pil().open(io.BytesIO(self.data))
        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGBA")
        return np.asarray(pil_image)


Predicate = Callable[[Any], bool]
Converter = Callable[[Any], FigureData]

_CONVERTERS: list[tuple[Predicate, Converter]] = []


def register_figure_converter(predicate: Predicate, converter: Converter) -> None:
    """Register a converter used by ``Logger.log_figure`` to turn objects into images.

    Converters registered later take precedence, so users can override the built-in ones.

    Args:
        predicate: Returns ``True`` if ``converter`` can handle the object.
        converter: Converts the object into a :class:`FigureData`.
    """
    _CONVERTERS.append((predicate, converter))


def to_figure_data(obj: Any) -> FigureData:
    """Convert a figure, image or array into an encoded image.

    Raises:
        TypeError: If there is no converter for the object type.
    """
    if isinstance(obj, FigureData):
        return obj

    for predicate, converter in reversed(_CONVERTERS):
        if predicate(obj):
            return converter(obj)

    raise TypeError(
        f"Don't know how to log a figure of type `{type(obj).__module__}.{type(obj).__qualname__}`."
        f" Use `reax.loggers.register_figure_converter()` to add support for it."
    )


def figure_path(name: str, step: int | None, fmt: str, sep: str = "/") -> str:
    """Get the relative (``/`` separated) path that a figure should be stored at."""
    parts = [part for part in name.split(sep) if part]
    if not parts:
        raise ValueError(f"Invalid figure name: {name!r}")

    if step is None:
        return "/".join(["figures", *parts]) + f".{fmt}"

    return "/".join(["figures", *parts, f"step_{step:08d}.{fmt}"])


def is_matplotlib_figure(obj: Any) -> bool:
    return _from_module(obj, "matplotlib") and hasattr(obj, "savefig")


def is_plotly_figure(obj: Any) -> bool:
    return _from_module(obj, "plotly") and hasattr(obj, "to_image")


def close_figure(obj: Any) -> None:
    """Close the figure (if it is something that can be closed, e.g. a matplotlib figure)."""
    if is_matplotlib_figure(obj):
        from matplotlib import pyplot as plt

        plt.close(obj)


def _from_module(obj: Any, module: str) -> bool:
    mod = type(obj).__module__ or ""
    return mod == module or mod.startswith(module + ".")


def _import_pil():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "`pillow` is required to log images from arrays, please install it with "
            "`pip install pillow`"
        ) from exc

    return Image


# region Built-in converters


def _matplotlib_to_png(fig: Any) -> FigureData:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return FigureData(buffer.getvalue(), "png")


def _plotly_to_png(fig: Any) -> FigureData:
    try:
        data = fig.to_image(format="png")
    except (ImportError, ValueError) as exc:
        raise ImportError(
            "Rendering plotly figures to PNG requires `kaleido`, please install it with "
            "`pip install kaleido`"
        ) from exc
    return FigureData(data, "png")


def _is_pil_image(obj: Any) -> bool:
    return _from_module(obj, "PIL") and hasattr(obj, "save") and hasattr(obj, "mode")


def _pil_to_png(img: Any) -> FigureData:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return FigureData(buffer.getvalue(), "png")


def _is_array(obj: Any) -> bool:
    if isinstance(obj, np.ndarray):
        return True
    # Check for JAX arrays without importing jax
    return _from_module(obj, "jax") or _from_module(obj, "jaxlib")


def _array_to_png(arr: Any) -> FigureData:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[-1] not in (3, 4)):
        raise ValueError(
            f"Image arrays must have shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got "
            f"{arr.shape}"
        )

    if np.issubdtype(arr.dtype, np.floating) or arr.dtype == np.bool_:
        arr = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return _pil_to_png(_import_pil().fromarray(arr))


def _is_encoded_image(obj: Any) -> bool:
    return isinstance(obj, (bytes, bytearray, memoryview))


def _encoded_to_figure_data(obj: Any) -> FigureData:
    data = bytes(obj)
    if data.startswith(PNG_MAGIC):
        return FigureData(data, "png")
    if data.startswith(JPEG_MAGIC):
        return FigureData(data, "jpeg")

    raise ValueError("Only PNG and JPEG encoded images are supported when passing raw bytes")


def _has_repr_png(obj: Any) -> bool:
    return callable(getattr(obj, "_repr_png_", None))


def _repr_png_to_figure_data(obj: Any) -> FigureData:
    data = obj._repr_png_()  # pylint: disable=protected-access
    if data is None:
        raise ValueError(f"`{type(obj).__name__}._repr_png_()` returned None")
    return FigureData(bytes(data), "png")


# Lowest priority first
register_figure_converter(_has_repr_png, _repr_png_to_figure_data)
register_figure_converter(_is_encoded_image, _encoded_to_figure_data)
register_figure_converter(_is_array, _array_to_png)
register_figure_converter(_is_pil_image, _pil_to_png)
register_figure_converter(is_plotly_figure, _plotly_to_png)
register_figure_converter(is_matplotlib_figure, _matplotlib_to_png)

# endregion

import re
import warnings

__all__ = ("no_warning_call",)


class no_warning_call:
    """Context manager asserting that no warning matching `expected_warning`/`match` is raised."""

    def __init__(self, expected_warning: type[Warning] = UserWarning, match: "str | None" = None):
        self.expected_warning = expected_warning
        self.match = match

    def __enter__(self):
        self._catcher = warnings.catch_warnings(record=True)
        self._record = self._catcher.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._catcher.__exit__(exc_type, exc_value, traceback)
        if exc_type is not None:
            return False

        for warning in self._record:
            if not issubclass(warning.category, self.expected_warning):
                continue
            if self.match is None or re.compile(self.match).search(str(warning.message)):
                raise AssertionError(
                    f"Warning matching {self.expected_warning} was raised: {warning.message}"
                )

        return False

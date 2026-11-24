import logging
import pathlib

import jax
from typing_extensions import override

from . import profiler

__all__ = ("JaxProfiler",)

_LOGGER = logging.getLogger(__name__)


class JaxProfiler(profiler.Profiler):
    def __init__(
        self,
        log_dir: str | pathlib.Path = ".",
        create_perfetto_link=False,
        create_perfetto_trace=False,
        profiler_options=None,
    ):
        # Params
        self._log_dir = log_dir
        self._create_perfetto_link = create_perfetto_link
        self._create_perfetto_trace = create_perfetto_trace
        self._profiler_options = profiler_options

        # State
        self._running: str | None = None
        self._traces: dict[str, jax.profiler.TraceAnnotation] = {}

    @override
    def start(self, profile_name: str, **_kwargs):
        """Start a new trace"""
        jax.profiler.start_trace(
            self._log_dir,
            create_perfetto_link=self._create_perfetto_link,
            create_perfetto_trace=self._create_perfetto_trace,
            profiler_options=self._profiler_options,
        )
        self._running = profile_name
        self.start_recording(profile_name)

    @override
    def stop(self, profile_name: str):
        """Stops an existing trace"""
        if self._running != profile_name:
            raise RuntimeError("Cannot stop, the profiler is not running")

        self.stop_recording(self._running)
        jax.profiler.stop_trace()
        self._running = None

    @override
    def start_recording(self, action_name: str) -> None:
        trace = jax.profiler.TraceAnnotation(action_name)
        trace.__enter__()
        self._traces[action_name] = trace

    @override
    def stop_recording(self, action_name: str) -> None:
        if trace := self._traces.pop(action_name, None):
            trace.__exit__(None, None, None)

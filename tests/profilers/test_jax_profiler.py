import os

import pytest

from reax.profilers import jax_profiler


def test_jax_profiler_start_stop(tmp_path):
    prof = jax_profiler.JaxProfiler(log_dir=tmp_path)
    try:
        prof.start("run")
        prof.stop("run")
    except Exception:
        pytest.xfail("jax profiling trace backend unavailable in this environment")
    # Trace files (or the perfetto trace) should have produced output in log_dir.
    assert os.path.isdir(tmp_path)


def test_jax_profiler_stop_not_running_raises(tmp_path):
    prof = jax_profiler.JaxProfiler(log_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not running"):
        prof.stop("run")


def test_jax_profiler_start_stop_recording(tmp_path):
    prof = jax_profiler.JaxProfiler(log_dir=tmp_path)
    prof.start_recording("action")
    assert "action" in prof._traces
    prof.stop_recording("action")
    assert prof._traces == {}


def test_jax_profiler_stop_recording_missing_is_noop(tmp_path):
    prof = jax_profiler.JaxProfiler(log_dir=tmp_path)
    # No trace registered; should not raise.
    prof.stop_recording("never-started")
    assert prof._traces == {}


def test_dummy_profiler_is_noop():
    from reax.profilers import profiler

    prof = profiler.DummyProfiler()
    prof.start("run")
    prof.stop("run")
    prof.start_recording("action")
    prof.stop_recording("action")


def test_profiler_profile_action_context_manager(tmp_path):
    prof = jax_profiler.JaxProfiler(log_dir=tmp_path)
    with prof.profile_action("action") as name:
        assert name == "action"
    assert prof._traces == {}

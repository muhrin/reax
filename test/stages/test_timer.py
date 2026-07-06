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
#
# Most of this file is covered by the following license.  To find what has been modified you
# can perform a diff with the file at:
# https://github.com/Lightning-AI/pytorch-lightning/blob/f9babd1def4c703e639dfc34fd1877ac4e7b9435/tests/tests_pytorch/callbacks/test_timer.py
#
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import timedelta
import logging
import time
from unittest.mock import Mock, patch

import pytest

import reax
from reax import demos, exceptions
from reax.stages import _timer


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        ("00:00:00:22", timedelta(seconds=22)),
        ("12:34:56:65", timedelta(days=12, hours=34, minutes=56, seconds=65)),
        (timedelta(weeks=52, milliseconds=1), timedelta(weeks=52, milliseconds=1)),
        ({"weeks": 52, "days": 1}, timedelta(weeks=52, days=1)),
    ],
)
def test_timer_parse_duration(duration, expected):
    timer = _timer.StageTimer(max_time=duration)
    assert (timer.time_remaining() == expected is None) or (
        timer.time_remaining() == expected.total_seconds()
    )


@pytest.mark.parametrize("duration", ["6:00:00", "60 minutes"])
def test_timer_parse_duration_misconfiguration(duration):
    with pytest.raises(exceptions.MisconfigurationException, match="format DD:HH:MM:SS"):
        _timer.StageTimer(max_time=duration)


@patch("reax.stages._timer.time")
def test_timer_time_remaining(time_mock):
    """Test that the timer tracks the elapsed and remaining time correctly."""
    start_time = time.monotonic()
    duration = timedelta(seconds=10)
    time_mock.monotonic.return_value = start_time
    timer = _timer.StageTimer(max_time=duration)
    assert timer.time_remaining() == duration.total_seconds()
    assert timer.time_elapsed() == 0

    # timer not started yet
    time_mock.monotonic.return_value = start_time + 60
    assert timer.start_time() is None
    assert timer.time_remaining() == 10
    assert timer.time_elapsed() == 0

    # start timer
    time_mock.monotonic.return_value = start_time
    timer.start()
    assert timer.start_time() == start_time

    # pretend time has elapsed
    elapsed = 3
    time_mock.monotonic.return_value = start_time + elapsed
    assert timer.start_time() == start_time
    assert round(timer.time_remaining()) == 7
    assert round(timer.time_elapsed()) == 3


def test_timer_stops_training(tmp_path, caplog):
    """Test that the timer stops training before reaching max_epochs."""
    model = demos.BoringModel()
    duration = timedelta(milliseconds=1000)

    trainer = reax.Trainer(default_root_dir=tmp_path)
    with caplog.at_level(logging.INFO):
        trainer.fit(model, max_epochs=1000, max_time=duration)
    assert trainer.global_updates >= 1
    assert trainer.current_epoch < 999
    assert "Time limit reached." in caplog.text
    assert "Signaling Trainer to stop." in caplog.text


def test_timer_zero_duration_stop(tmp_path):
    """Test that the timer stops training immediately after the first check occurs."""
    model = demos.BoringModel()
    duration = timedelta(0)
    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(model, max_time=duration)
    assert trainer.global_updates == 0
    assert trainer.current_epoch == 0


@pytest.mark.parametrize(("min_steps", "min_epochs"), [(0, 2), (3, 0), (3, 2)])
def test_timer_duration_min_steps_override(tmp_path, min_steps, min_epochs):
    model = demos.BoringModel()
    duration = timedelta(0)
    trainer = reax.Trainer(default_root_dir=tmp_path)
    fit = trainer.fit(model, min_updates=min_steps, min_epochs=min_epochs, max_time=duration)
    if min_epochs:
        assert trainer.current_epoch >= min_epochs
    if min_steps:
        assert trainer.global_updates >= min_steps - 1
    assert fit.timer.time_elapsed() > duration.total_seconds()

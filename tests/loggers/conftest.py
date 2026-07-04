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
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/tests/tests_pytorch/loggers/conftest.py
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
import sys
import types
from unittest import mock

import pytest


@pytest.fixture()
def mlflow_mock(monkeypatch):
    mlflow = types.ModuleType("mlflow")
    mlflow.set_tracking_uri = mock.Mock()
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)

    mlflow_tracking = types.ModuleType("tracking")
    mlflow_tracking.MlflowClient = mock.Mock()
    mlflow_tracking.artifact_utils = mock.Mock()
    monkeypatch.setitem(sys.modules, "mlflow.tracking", mlflow_tracking)

    mlflow_entities = types.ModuleType("entities")
    mlflow_entities.Metric = mock.Mock()
    mlflow_entities.Param = mock.Mock()
    mlflow_entities.time = mock.Mock()
    monkeypatch.setitem(sys.modules, "mlflow.entities", mlflow_entities)

    mlflow.tracking = mlflow_tracking
    mlflow.entities = mlflow_entities

    monkeypatch.setattr("reax.loggers.mlflow._MLFLOW_AVAILABLE", True)
    monkeypatch.setattr("reax.loggers.mlflow._MLFLOW_SYNCHRONOUS_AVAILABLE", True)
    return mlflow

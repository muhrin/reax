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
# https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/61a8391617b3f2b5083852ea49b1ead4c2ef72d7/src/lightning/fabric/utilities/rank_zero.py
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
"""Utilities that can be used for calling functions on a particular rank."""

import logging
import os
from typing import Optional

import lightning_utilities.core.rank_zero as rank_zero_module

# note: we want to keep these indirections so the `rank_zero_only.rank` is set on import
from lightning_utilities.core.rank_zero import (  # noqa: F401
    WarningCache,
    rank_prefixed_message,
    rank_zero_debug,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from typing_extensions import ParamSpec

rank_zero_module.log = logging.getLogger(__name__)


def _get_rank() -> Optional[int]:
    """Get rank."""
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


rank_zero_only = rank_zero_module.rank_zero_only
# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank() or 0)


class ReaxDeprecationWarning(DeprecationWarning):
    """Deprecation warnings raised by Lightning."""


rank_zero_module.rank_zero_deprecation_category = ReaxDeprecationWarning

# Copyright (C) 2025  Martin Uhrin
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
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/src/lightning/pytorch/utilities/grads.py
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
"""Utilities to describe gradients."""

from typing import Any, Union

import jax.numpy as jnp
from pytray import tree

__all__ = ("grad_norm",)


def grad_norm(
    grads: dict[str, Any], norm_type: Union[float, int, str], group_separator: str = "/"
) -> dict[str, float]:
    """Compute each parameter's gradient's norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        grads: the gradients PyTree, typically coming from a ``.grad()`` call
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.

    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(
            f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}"
        )

    def path_name(path) -> str:
        """Given a tree path this will return a string version of the path"""
        path_str = tree.path_to_string(path).replace(" ", "_")
        return f"grad_{norm_type}_norm{group_separator}{path_str}"

    norms = {
        path_name(path): jnp.linalg.norm(value.flatten(), ord=norm_type)
        for path, value in tree.flatten(grads).items()
        if value is not None
    }

    if norms:
        total_norm = jnp.linalg.norm(jnp.array(list(norms.values())), ord=norm_type)
        norms[f"grad_{norm_type}_norm_total"] = total_norm
    return norms

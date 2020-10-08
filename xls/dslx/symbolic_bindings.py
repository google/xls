# Lint as: python3
#
# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared type definition for symbolic bindings.

For example, in:

  fn [M: u32, N: u32] f(x: bits[M], y: bits[N]) -> bits[N] {
    ...
  }

The symbolic bindings that instantiate this function might be:

  (('M', 42), ('N', 64))
"""

from typing import Tuple

SymbolicBindings = Tuple[Tuple[str, int], ...]

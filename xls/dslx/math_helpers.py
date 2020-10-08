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

"""Helper routines dealing with mathematical operations."""

import functools
import operator
from typing import Iterable, TypeVar, Optional


def prod(xs: Iterable[TypeVar('T')],
         one: Optional[TypeVar('T')]=None) -> TypeVar('T'):
  """Returns the product of 'xs'.

  Args:
    xs: Iterable to take the product of.
    one: If provided and the iterable is empty, this value is returned.

  Raises:
    ValueError: If no 'one' value is provided and an empty iterator is given.
  """
  accum = functools.reduce(operator.mul, xs, one)
  if accum is None:
    raise ValueError(
        'Empty iterator provided to prod() and no "one" value given.')
  return accum

# Lint as: python3
#
# Copyright 2020 Google LLC
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

# TODO(leary): 2020-09-06 Rename file to positional_error.py.
"""Base error for positional errors in the source text."""

from xls.common.xls_error import XlsError
from xls.dslx.python import cpp_pos


class PositionalError(XlsError):
  """An XLS error that's associated with a span position in source text."""

  def __init__(self, message: str, span: cpp_pos.Span):
    super(PositionalError, self).__init__(message)
    self.span = span
    self.printed = False

  @property
  def filename(self) -> str:
    return self.span.filename

  @property
  def message(self) -> str:
    return self.args[0]

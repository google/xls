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

"""Positional parse error class for errors during parsing."""

from xls.dslx.python import cpp_pos
from xls.dslx.span import PositionalError


class ParseError(PositionalError):
  """Raised when there is a grammatical error in the input program.

  Attributes:
    span: Span in the text at which the parse error occurred.
  """

  def __init__(self, span: cpp_pos.Span, message: str):
    super(ParseError, self).__init__('{} @ {}'.format(message, span), span)
    self.span = span

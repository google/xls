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

"""Error types used during syntax interpretations."""

from xls.dslx.python import cpp_pos
from xls.dslx.span import PositionalError


class FailureError(PositionalError):
  """Raised when expression evaluation fails (as in 'panic' style failure).

  This is used e.g. in tests, but may be reusable for things like fatal errors.
  """

  def __init__(self, span: cpp_pos.Span, message: str):
    super(FailureError, self).__init__(message, span)


class EvaluateError(PositionalError):
  """Raised when an error is encountered during interpreted evaluation.

  For example, if no type checking is performed ahead of time and the
  interpreter gets incompatible types to an operation, an EvaluateError will be
  raised.
  """

  def __init__(self, span: cpp_pos.Span, message: str):
    super(EvaluateError, self).__init__(message, span)


class InstantiationError(PositionalError):
  """Raised when there is an issue with parametric instantiation.

  For example if you instantiate the following with different bit-width for x
  and y:

    fn parametric(x: bits[N], y: bits[N])

  This error will be raised.
  """

  def __init__(self, span: cpp_pos.Span, message: str):
    super(InstantiationError, self).__init__(message, span)

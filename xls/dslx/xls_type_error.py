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

# Lint as: python3
"""Defines the type error that DSLX can produce during type checking."""

from typing import Text, Optional, Any, Tuple

from xls.dslx.concrete_type import ConcreteType
from xls.dslx.span import PositionalError
from xls.dslx.span import Span


class XlsTypeError(PositionalError):
  """Error that is raised when there is a type checking error for DSLX code."""

  def __init__(self,
               span: Span,
               lhs_type: Optional[ConcreteType],
               rhs_type: Optional[ConcreteType],
               suffix: Text = ''):
    msg = ('Types are not compatible: {0} vs {1}{suffix} @ {span}').format(
        '<none>' if lhs_type is None else lhs_type,
        '<none>' if rhs_type is None else rhs_type,
        span=span,
        suffix=': ' + suffix if suffix else '')
    super(XlsTypeError, self).__init__(msg, span)
    # TODO(leary): 2019-01-22 Break out AST into its own module so we can
    # annotate these as types without circular dependency between parser and
    # type error.
    self.lhs_type = lhs_type
    self.rhs_type = rhs_type
    self.suffix = suffix


class TypeInferenceError(PositionalError):
  """Error raised when an error occurs during deductive type inference.

  Attributes:
    span: The span at which the type deduction error occurred.
    type_: The (AST) type that failed to deduce. We avoid annotating the
      ast.Type here to break a circular dependency.
    suffix: Message suffix to use when displaying the error.
  """

  def __init__(self,
               span: Span,
               type_: Optional[Any] = None,
               suffix: Text = ''):
    msg = 'Could not infer type{}{} @ {}'.format(
        ' for {}'.format(type_) if type_ else '',
        ': ' + suffix if suffix else '', span)
    super(TypeInferenceError, self).__init__(msg, span)
    self.type_ = type_
    self.suffix = suffix


class ArgCountMismatchError(PositionalError):
  """Raised when argument count != parameter count in an invocation."""

  def __init__(self, span: Span, arg_types: Tuple[ConcreteType,
                                                  ...], param_count: int,
               param_types: Optional[Tuple[ConcreteType,
                                           ...]], suffix: Optional[Text]):
    self.arg_types = arg_types
    self.param_types = param_types
    parameters_str = ' parameters: [{}];'.format(', '.join(
        str(p) for p in param_types)) if param_types else ''
    msg = ('Expected {} parameter(s) but got {} argument(s);{} '
           'arguments: [{}]').format(param_count, len(arg_types),
                                     parameters_str,
                                     ', '.join(str(a) for a in arg_types))
    super(ArgCountMismatchError, self).__init__(msg, span)

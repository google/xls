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

"""Parametric dimension "sub-AST".

After we resolve type references in the AST we get "concrete types" where we
can "see through" all typerefs and there's a clear distinction between bits and
tuples. However, parametric functions are still symbolic at this point. As a
result, we have `ConcreteType[ParametricExpression]` hold the parametric
dimensions in bits types. At IR conversion time, when all dimensions are
apparent as the flow down from the "main" function,
`ConcreteType[ParametricExpression]` can all be evaluated to `ConcreteType[int]`
via `ParametricExpression.evaluate()`.
"""

import abc
from typing import Text, Dict, Set, Union

from xls.dslx.span import Span


class ParametricExpression(object):  # pytype: disable=ignored-metaclass
  """Abstract base class for a parametric dimension expression.

  Parametric dimension expressions can be evaluated with an environment. For
  example, with the parametric type:

    bits[M+N]

  If we evalute the parametric expression M+N with an environment {M: 3, N; 7}
  we get:

    bits[10]
  """

  __metaclass__ = abc.ABCMeta

  def __add__(self, other: 'ParametricExpression') -> 'ParametricAdd':
    assert isinstance(other, ParametricExpression), other
    return ParametricAdd(self, other)

  @abc.abstractmethod
  def __eq__(self, other: 'ParametricExpression'):
    raise NotImplementedError

  def __ne__(self, other: 'ParametricExpression'):
    return not self.__eq__(other)

  def __radd__(self, lhs: int) -> 'ParametricExpression':
    assert isinstance(lhs, int), lhs
    if lhs == 0:
      return self
    return ParametricAdd(ParametricConstant(lhs), self)

  def __rmul__(self, lhs: int) -> 'ParametricExpression':
    assert isinstance(lhs, int), lhs
    if lhs == 1:
      return self
    return ParametricMul(ParametricConstant(lhs), self)

  def __mul__(self, rhs: int) -> 'ParametricExpression':
    return self.__rmul__(rhs)  # Symmetric.

  @abc.abstractmethod
  def get_freevars(self) -> Set[Text]:
    raise NotImplementedError

  @abc.abstractmethod
  def evaluate(self, env: Dict[Text,
                               int]) -> Union[int, 'ParametricExpression']:
    raise NotImplementedError


class ParametricConstant(ParametricExpression):
  """Represents a constant value in a parametric dimension expression.

  For example, when you do:

    bits[1]:0 ++ bits[N]:x

  It produces a parametric expression for the type:

    bits[1+N]

  Where the '1' is a parametric constant.

  Attributes:
    value: The constant value.
  """

  def __init__(self, value: int):
    self.value = value

  def __eq__(self, other: ParametricExpression) -> bool:
    return isinstance(other, ParametricConstant) and self.value == other.value

  def __str__(self) -> Text:
    return str(self.value)

  def get_freevars(self) -> Set[Text]:
    return set()

  def evaluate(self, env: Dict[Text, int]) -> Union[int, ParametricExpression]:
    return self.value


class ParametricMul(ParametricExpression):
  """Represents a multiplication in a parametric dimension expression."""

  def __init__(self, lhs: ParametricExpression, rhs: ParametricExpression):
    self.lhs = lhs
    self.rhs = rhs

  def __eq__(self, other: ParametricExpression) -> bool:
    return isinstance(
        other,
        ParametricMul) and self.lhs == other.lhs and self.rhs == other.rhs

  def __repr__(self) -> Text:
    return 'ParametricMul({!r}, {!r})'.format(self.lhs, self.rhs)

  def __str__(self) -> Text:
    return '({})*({})'.format(self.lhs, self.rhs)

  def get_freevars(self) -> Set[Text]:
    return self.lhs.get_freevars() | self.rhs.get_freevars()

  def evaluate(self, env: Dict[Text, int]) -> Union[int, ParametricExpression]:
    return self.lhs.evaluate(env) * self.rhs.evaluate(env)


class ParametricAdd(ParametricExpression):
  """Represents an add in a parametric dimension expression."""

  def __init__(self, lhs: ParametricExpression, rhs: ParametricExpression):
    self.lhs = lhs
    self.rhs = rhs

  def __eq__(self, other: ParametricExpression) -> bool:
    return isinstance(
        other,
        ParametricAdd) and self.lhs == other.lhs and self.rhs == other.rhs

  def __str__(self) -> Text:
    return '({})+({})'.format(self.lhs, self.rhs)

  def __repr__(self) -> Text:
    return 'ParametricAdd({!r}, {!r})'.format(self.lhs, self.rhs)

  def get_freevars(self) -> Set[Text]:
    return self.lhs.get_freevars() | self.rhs.get_freevars()

  def evaluate(self, env: Dict[Text, int]) -> Union[int, ParametricExpression]:
    return self.lhs.evaluate(env) + self.rhs.evaluate(env)


class ParametricSub(ParametricExpression):
  """Represents an add in a parametric dimension expression."""

  def __init__(self, lhs: ParametricExpression, rhs: ParametricExpression):
    self.lhs = lhs
    self.rhs = rhs

  def __eq__(self, other: ParametricExpression) -> bool:
    return isinstance(
        other,
        ParametricSub) and self.lhs == other.lhs and self.rhs == other.rhs

  def __str__(self) -> Text:
    return '({})-({})'.format(self.lhs, self.rhs)

  def __repr__(self) -> Text:
    return 'ParametricSub({!r}, {!r})'.format(self.lhs, self.rhs)

  def get_freevars(self) -> Set[Text]:
    return self.lhs.get_freevars() | self.rhs.get_freevars()

  def evaluate(self, env: Dict[Text, int]) -> Union[int, ParametricExpression]:
    return self.lhs.evaluate(env) - self.rhs.evaluate(env)


class ParametricSymbol(ParametricExpression):
  """Represents a symbol in a parametric dimension expression.

  For example, in the expression:

    bits[M+N+1]

  Both M and N are parametric symbols.

  Attributes:
    identifier: The text identifier for the parametric symbol.
    span: Span in the source text where this parametric symbol resides.
  """

  def __init__(self, identifier: Text, span: Span):
    self.identifier = identifier
    self.span = span

  def __repr__(self) -> Text:
    return 'ParametricSymbol({!r})'.format(self.identifier)

  def __eq__(self, other: ParametricExpression) -> bool:
    return isinstance(other,
                      ParametricSymbol) and self.identifier == other.identifier

  def __str__(self) -> Text:
    return self.identifier

  def get_freevars(self) -> Set[Text]:
    return set([self.identifier])

  def evaluate(self, env: Dict[Text, int]) -> Union[int, ParametricExpression]:
    try:
      return env[self.identifier]
    except KeyError:
      return self

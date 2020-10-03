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
"""(Python) helper routines for working with (C++) ConcreteType objects."""

import functools
from typing import Union, Callable

from xls.dslx.python import cpp_ast as ast
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import EnumType
from xls.dslx.python.cpp_concrete_type import FunctionType
from xls.dslx.python.cpp_concrete_type import TupleType

ConcreteType.NIL = TupleType(())
Dim = Union[str, int]


def map_size(t: ConcreteType, m: ast.Module, f: Callable[[Dim],
                                                         Dim]) -> ConcreteType:
  """Runs f on all dimensions within t (transively for contained types)."""
  assert isinstance(m, ast.Module), m
  rec = functools.partial(map_size, m=m, f=f)

  if isinstance(t, ArrayType):
    return ArrayType(rec(t.get_element_type()), f(t.size))
  elif isinstance(t, BitsType):
    return BitsType(t.signed, f(t.size))
  elif isinstance(t, TupleType):
    nominal = t.get_nominal_type(m)
    if t.named:
      return TupleType(
          tuple((name, rec(type_)) for name, type_ in t.members), nominal)
    assert nominal is None, nominal
    return TupleType(tuple(rec(e) for e in t.members))
  elif isinstance(t, EnumType):
    return EnumType(t.get_nominal_type(m), f(t.size))
  elif isinstance(t, FunctionType):
    mapped_params = tuple(rec(p) for p in t.params)
    mapped_return_type = rec(t.return_type)
    return FunctionType(mapped_params, mapped_return_type)
  else:
    raise NotImplementedError(t.__class__)

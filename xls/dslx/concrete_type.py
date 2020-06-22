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
"""'Concrete' types are fully instantiated / known types.

Types with parameters are not 'concrete', they're parametric -- when they
resolve, they resolve to concrete types.

Concrete types are used in both type inference deduction and interpreter
evaluation of the DSL.
"""

import abc
from typing import Optional, Tuple, Text, Generic, TypeVar, Callable, Any, Union, cast

from absl import logging


DimsT = TypeVar('DimsT')
DimsU = TypeVar('DimsU')
NominalType = Any  # E.g. ast.Enum or ast.Struct.


class ConcreteType(Generic[DimsT], metaclass=abc.ABCMeta):
  """Represents a 'concrete' (fully evaluated) type.

  Type constructs in the AST may have abstract expressions e.g. in the
  dimensions fields, but ultimately they resolve into some concrete type when
  those are evaluated. This class represents that resolved result.

  During typechecking the dimension members may be either symbols (like the 'N'
  in bits[N,3]) or integers, i.e. ConcreteType[Union[int, Text]]. Once
  parametric symbols are instantiated the symbols (such as 'N') will have
  resolved into ints, and we will only be dealing with ConcreteType[int].
  """

  def compatible_with(self, other: 'ConcreteType') -> bool:
    """Type equality, but ignores tuple member naming discrepancies."""
    if self == other:  # Equality implies compatibility.
      return True

    if isinstance(self, TupleType) and isinstance(other, TupleType):
      for s, o in zip(self.get_unnamed_members(), other.get_unnamed_members()):
        if not s.compatible_with(o):
          logging.vlog(5, '%s not compatible with %s', s, o)
          return False
      return True

    if isinstance(self, ArrayType) and isinstance(other, ArrayType):
      result = self.get_element_type().compatible_with(other.get_element_type())
      logging.vlog(5, '%s compatible with %s => %s', self, other, result)
      return result

    return False

  @abc.abstractmethod
  def __eq__(self, other: 'ConcreteType[DimsT]') -> bool:
    raise NotImplementedError

  def __ne__(self, other: 'ConcreteType[DimsT]') -> bool:
    return not self.__eq__(other)

  @abc.abstractmethod
  def get_debug_type_name(self) -> str:
    """Returns name for the type of this object suitable for error messages."""
    raise NotImplementedError

  @abc.abstractmethod
  def map_size(self, f: Callable[[DimsT], DimsU]) -> 'ConcreteType[DimsU]':
    """Applies f to all sizes in "self" recursively and returns the new type."""
    raise NotImplementedError

  @abc.abstractmethod
  def has_enum(self) -> bool:
    """Returns true if this concrete type has an enum member (recursively)."""
    raise NotImplementedError

  def is_nil(self) -> bool:
    """Returns true if this is an empty tuple (AKA 'nil' tuple)."""
    return isinstance(self, TupleType) and not self.get_tuple_length()

  @abc.abstractmethod
  def get_all_dims(self) -> Tuple[DimsT, ...]:
    raise NotImplementedError

  @abc.abstractmethod
  def get_total_bit_count(self) -> DimsT:
    raise NotImplementedError


class EnumType(ConcreteType[DimsT]):
  """Represents an enum type."""

  def __init__(self, enum: NominalType, bit_count: DimsT):
    assert enum is not None
    self.nominal_type = enum
    self.size = bit_count

  def __str__(self) -> str:
    return self.nominal_type.identifier

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, EnumType):
      return False
    return self.nominal_type == other.nominal_type and self.size == other.size

  def get_signedness(self) -> bool:
    return self.nominal_type.get_signedness()

  def get_total_bit_count(self) -> DimsT:
    return self.size

  def get_all_dims(self) -> Tuple[DimsT, ...]:
    return (self.size,)

  def has_enum(self) -> bool:
    return True

  def get_debug_type_name(self) -> str:
    return 'enum'

  def map_size(self, f: Callable[[DimsT], DimsU]) -> 'ConcreteType[DimsU]':
    """Applies f to all sizes in "self" recursively and returns the new type."""
    return EnumType(self.nominal_type, f(self.size))


# Tuple members are either named or unnamed.
UnnamedTupleTypeMembers = Tuple[ConcreteType[DimsT], ...]
NamedTupleTypeMembers = Tuple[Tuple[str, ConcreteType[DimsT]], ...]
TupleTypeMembers = Union[UnnamedTupleTypeMembers, NamedTupleTypeMembers]  # pytype: disable=not-supported-yet


class TupleType(ConcreteType[DimsT]):
  """Represents a tuple type.

  Tuples can have unnamed members or named members. In any case, you can request
  `tuple_type.get_unnamed_members()`.

  When the members are named the `nominal_type` may refer to the struct
  definition that led to those named members.
  """

  def __init__(self,
               members: TupleTypeMembers,
               struct: Optional[NominalType] = None):
    self._members = members
    self.nominal_type = struct

  def __eq__(self, other: Any) -> bool:
    """Returns whether this tuple type is exactly equal to other tuple type."""
    if not isinstance(other, TupleType):
      return False
    return (self._members == other._members and
            self.nominal_type == other.nominal_type)

  def __str__(self) -> str:

    def member_to_string(m) -> Text:
      if isinstance(m, tuple):
        return '{}: {}'.format(m[0], m[1])
      return str(m)

    return '({}{})'.format(
        ', '.join(member_to_string(m) for m in self._members),
        ',' if len(self._members) == 1 else '')

  def get_debug_type_name(self) -> str:
    return 'tuple'

  def has_enum(self) -> bool:
    return any(m.has_enum() for m in self.get_unnamed_members())

  def _get_named_members(self) -> NamedTupleTypeMembers:
    assert all(isinstance(item, tuple) for item in self._members), \
        'Precondition: members of tuple must be named: {}'.format(self)
    return cast(Any, self._members)

  def get_total_bit_count(self) -> DimsT:
    return sum(m.get_total_bit_count() for m in self.get_unnamed_members())

  def get_unnamed_members(self) -> UnnamedTupleTypeMembers:
    result = []
    for m in self._members:
      if isinstance(m, tuple):
        result.append(m[1])
      else:
        result.append(m)
    return tuple(result)

  def get_all_dims(self) -> Tuple[DimsT, ...]:
    all_dims = []
    for m in self.get_unnamed_members():
      d = m.get_all_dims()
      assert isinstance(d, tuple), d
      all_dims += d
    return tuple(all_dims)

  def get_member_type_by_name(self, target: Text) -> ConcreteType[DimsT]:
    for k, v in self._get_named_members():
      if k == target:
        return v
    raise KeyError('Cannot find member with name: {!r}'.format(target))

  def get_tuple_length(self) -> int:
    return len(self._members)

  @property
  def tuple_names(self) -> Tuple[Text, ...]:
    return tuple(k for k, _ in self._get_named_members())

  def has_named_member(self, target: Text) -> bool:
    return any(k == target for k, _ in self._get_named_members())

  def get_named_member_type(self, target: Text) -> ConcreteType[DimsT]:
    return next(t for k, t in self._get_named_members() if k == target)

  def get_tuple_member(self, i: int) -> ConcreteType[DimsT]:
    """Returns the ith tuple member of this concrete tuple type."""
    return self.get_unnamed_members()[i]

  def map_size(self, f: Callable[[DimsT], DimsU]) -> ConcreteType[DimsU]:
    """Applies f to all sizes in "self" recursively and returns the new type."""
    mapped = []
    for m in self._members:
      if isinstance(m, tuple):
        k, v = m
        mapped.append((k, v.map_size(f)))
      else:
        mapped.append(m.map_size(f))
    return TupleType(tuple(mapped), self.nominal_type)


class BitsType(ConcreteType[DimsT]):
  """Represents a bits type (either signed or unsigned).

  Note there are related helpers `is_ubits()` and `is_sbits()` for concisely
  testing whether a `ConcreteType` is unsigned or signed BitsType, respectively.
  """

  def __init__(self, signed: bool, size: DimsT):
    self.signed = signed
    self.size = size

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, BitsType):
      return False
    return self.signed == other.signed and self.size == other.size

  def __str__(self) -> str:
    primitive = 'sN' if self.signed else 'uN'
    return '{}[{}]'.format(primitive, self.size)

  def get_debug_type_name(self) -> str:
    return 'sbits' if self.signed else 'ubits'

  def get_signedness(self) -> bool:
    return self.signed

  def has_enum(self) -> bool:
    return False  # BitsType contains no other type.

  def get_all_dims(self) -> Tuple[DimsT]:
    return (self.size,)

  def get_total_bit_count(self) -> DimsT:
    return self.size

  def to_ubits(self) -> 'BitsType[DimsT]':
    return BitsType(signed=False, size=self.size)

  def map_size(self, f: Callable[[DimsT], DimsU]) -> ConcreteType[DimsU]:
    """Applies f to all sizes in "self" recursively and returns the new type."""
    return BitsType(self.signed, f(self.size))


class FunctionType(ConcreteType[DimsT]):
  """Represents a function type."""

  def __init__(self, params: Tuple[ConcreteType[DimsT], ...],
               return_type: ConcreteType[DimsT],
               body_type: Optional[ConcreteType[DimsT]] = None):
    """We optionally track the body_type in case we want to check it later
    (e.g. in the case of parametric function instantiation)
    """
    assert params is not None
    assert return_type is not None
    self.params = params
    self.return_type = return_type
    self.body_type = body_type if body_type else return_type

  def __str__(self) -> str:
    return '({}) -> {}'.format(', '.join(str(p) for p in self.params),
                               self.return_type)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, FunctionType):
      return False
    return self.params == other.params and self.return_type == other.return_type

  def get_debug_type_name(self) -> str:
    return 'function'

  def get_total_bit_count(self) -> int:
    return sum(p.get_total_bit_count()
               for p in self.params) + self.return_type.get_total_bit_count()

  def get_function_params(self) -> Tuple[ConcreteType[DimsT], ...]:
    return self.params

  def get_function_return_type(self) -> ConcreteType[DimsT]:
    return self.return_type

  def get_function_body_type(self) -> ConcreteType[DimsT]:
    return self.body_type

  def get_all_dims(self) -> Tuple[DimsT, ...]:
    return sum(p.get_all_dims()
               for p in self.params) + self.return_type.get_all_dims()

  def has_enum(self) -> bool:
    return any(p.has_enum() for p in self.params) or self.return_type.has_enum()

  def map_size(self, f: Callable[[DimsT], DimsU]) -> 'ConcreteType[DimsU]':
    """Applies f to all sizes in "self" recursively and returns the new type."""
    new_params = tuple(p.map_size(f) for p in self.params)
    new_return_type = self.return_type.map_size(f)
    return FunctionType(new_params, new_return_type)


class ArrayType(ConcreteType[DimsT]):
  """Represents an array type, with an element type and size.

  These will nest in the case of multidimensional arrays.
  """

  def __init__(self, element_type: ConcreteType[DimsT], size: DimsT):
    assert element_type is not None
    self.size = size
    self.element_type = element_type

  def __str__(self) -> str:
    return '{}[{}]'.format(self.get_element_type(), self.size)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, ArrayType):
      return False
    return self.size == other.size and self.element_type == other.element_type

  def get_debug_type_name(self) -> str:
    return 'array'

  def get_element_type(self) -> ConcreteType[DimsT]:
    return self.element_type

  def get_all_dims(self) -> Tuple[DimsT, ...]:
    return (self.size,) + self.element_type.get_all_dims()

  def get_total_bit_count(self) -> DimsT:
    return self.element_type.get_total_bit_count() * self.size

  def has_enum(self) -> bool:
    return self.element_type.has_enum()

  def map_size(self, f: Callable[[DimsT], DimsU]) -> 'ConcreteType[DimsU]':
    """Applies f to all sizes in "self" recursively and returns the new type."""
    return ArrayType(self.element_type.map_size(f), f(self.size))


ConcreteType.U1 = BitsType(signed=False, size=1)
ConcreteType.U8 = BitsType(signed=False, size=8)
ConcreteType.U32 = BitsType(signed=False, size=32)
ConcreteType.U64 = BitsType(signed=False, size=64)
ConcreteType.S32 = BitsType(signed=True, size=32)
ConcreteType.S64 = BitsType(signed=True, size=64)
ConcreteType.NIL = TupleType((), None)


def is_ubits(t: ConcreteType) -> bool:
  return isinstance(t, BitsType) and not t.signed


def is_sbits(t: ConcreteType) -> bool:
  return isinstance(t, BitsType) and t.signed

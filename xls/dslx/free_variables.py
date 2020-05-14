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
"""Tracking of free variables references."""

import pprint
from typing import Any, Sequence, Union, List, Tuple, Optional, Dict, Text, Callable, Set
from xls.dslx.concrete_type import ConcreteType

# pylint: disable=invalid-name
NameRef = Any
NameDef = Any
BuiltinNameDef = Any
NodeToType = Any
AnyNameDef = Union[NameDef, BuiltinNameDef]
# pylint: enable=invalid-name


class FreeVariables(object):
  """Data structure that holds free variable reference data.

  Has a union() operator so that expressions can build up their free variable
  references as part of lexical analysis.
  """

  def __init__(self, values: Optional[Dict[Text, NameRef]] = None):
    self._values = values or {}  # type: Dict[Text, List[NameRef]]

  def __repr__(self) -> Text:
    return 'FreeVariables(values={!r})'.format(self._values)

  def __len__(self) -> int:
    """Returns the number of free variable names."""
    return len(self._values)

  def __str__(self) -> Text:
    return pprint.pformat(self._values)

  def _refs_to_def(self, refs: Sequence[NameRef]) -> AnyNameDef:
    return next(iter(refs)).name_def

  def keys(self) -> Set[Text]:
    return set(self._values.keys())

  def drop_defs(self, should_drop: Callable[[NameDef],
                                            bool]) -> 'FreeVariables':
    return FreeVariables({
        name: refs
        for name, refs in self._values.items()
        if not should_drop(self._refs_to_def(refs))
    })

  def get_name_def_tups(self) -> List[Tuple[Text, AnyNameDef]]:
    """Returns a list of (name, name_def) tuples."""
    return [(name, self._refs_to_def(refs))
            for (name, refs) in sorted(self._values.items())]

  def get_name_defs(self) -> List[AnyNameDef]:
    return [
        self._refs_to_def(refs) for (_, refs) in sorted(self._values.items())
    ]

  def get_name_type_pairs(self, node_to_type: NodeToType
                         ) -> List[Tuple[Text, ConcreteType]]:
    return [(name, node_to_type[name_def])
            for (name, name_def) in self.get_name_def_tups()]

  def union(self, other: 'FreeVariables') -> 'FreeVariables':
    """Returns the union of the references in self with those in other."""
    # PyLint doesn't realize we're accessing private members of the same class.
    # pylint: disable=protected-access
    result = FreeVariables(dict((k, list(v)) for k, v in self._values.items()))
    for key, refs in other._values.items():
      if key in result._values:
        result._values[key] += refs
      else:
        result._values[key] = list(refs)
    return result
    # pylint: enable=protected-access

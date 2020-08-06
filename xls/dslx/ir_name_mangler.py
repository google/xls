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
"""DSLX name mangling for XLS IR conversion."""

from typing import Tuple, Text, Optional, Set
from xls.dslx import ast

SymbolicBindings = Tuple[Tuple[Text, int], ...]


def mangle_dslx_name(function_name: Text, free_keys: Set[Text], m: ast.Module,
                     symbolic_bindings: Optional[SymbolicBindings]) -> Text:
  """Returns mangled name of function 'name' w/ given parametric bindings."""
  symbolic_binding_keys = set(k for k, _ in symbolic_bindings or ())
  if free_keys > symbolic_binding_keys:
    raise ValueError('Not enough symbolic bindings to convert function {!r}; '
                     'need {!r} got {!r}'.format(function_name, free_keys,
                                                 symbolic_binding_keys))
  mod_name = m.name.replace('.', '_')
  if not symbolic_bindings:
    return '__{}__{}'.format(mod_name, function_name)
  suffix = '_'.join(str(v) for _, v in symbolic_bindings)
  return '__{}__{}__{}'.format(mod_name, function_name, suffix)

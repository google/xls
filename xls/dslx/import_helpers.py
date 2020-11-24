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
"""Helpers for importing."""

from typing import Tuple, Optional

from xls.dslx import typecheck
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python import import_routines


class Importer:
  """Wraps a cache with a callable interface that performs import.

  Implementation note: the do_import routine requires a typecheck lambda to
  break a circular dependency, so this object takes a dependency on both type
  checking and the import_routines and provides that lambda.
  """

  def __init__(self):
    self.cache: Optional[import_routines.ImportCache] = \
        import_routines.ImportCache()

  def typecheck(self, module: ast.Module) -> type_info_mod.TypeInfo:
    return typecheck.check_module(module, f_import=self)

  def __call__(self, subject: Tuple[str, ...]) -> import_routines.ModuleInfo:
    assert isinstance(subject, tuple), subject
    subject = import_routines.ImportTokens(subject)
    return import_routines.do_import(self.typecheck, subject, self.cache)

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

"""Implementation of type checking functionality on a parsed AST object."""

from typing import Optional, Text, Tuple, Callable

from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_type_info as type_info
from xls.dslx.python import cpp_typecheck


ImportFn = Callable[[Tuple[Text, ...]], Tuple[ast.Module, type_info.TypeInfo]]


def check_module(module: ast.Module,
                 f_import: Optional[ImportFn]) -> type_info.TypeInfo:
  """Validates type annotations on all functions within "module".

  Args:
    module: The module to type check functions for.
    f_import: Callback to import a module (a la a import statement). This may be
      None e.g. in unit testing situations where it's guaranteed there will be
      no import statements.
  Returns:
    Mapping from AST node to its deduced/checked type.
  Raises:
    XlsTypeError: If any of the function in f have typecheck errors.
  """
  assert f_import is None or callable(f_import), f_import
  import_cache = None if f_import is None else getattr(f_import, 'cache')
  additional_search_paths = () if f_import is None else getattr(
      f_import, 'additional_search_paths')
  return cpp_typecheck.check_module(module, import_cache,
                                    additional_search_paths)

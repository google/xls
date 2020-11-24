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

"""Helper utilities for use with the interpreter.Interpreter."""

from typing import Tuple, Dict, Callable
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python import import_routines
from xls.dslx.python import interpreter
from xls.dslx.python.interp_bindings import FnCtx

SymbolicBindings = Tuple[Tuple[str, int], ...]


def interpret_expr(module: ast.Module, type_info: type_info_mod.TypeInfo,
                   typecheck: Callable[[ast.Module], type_info_mod.TypeInfo],
                   import_cache: import_routines.ImportCache, expr: ast.Expr, *,
                   env: Dict[str, int], bit_widths: Dict[str, int],
                   fn_ctx: Tuple[str, str, SymbolicBindings]) -> int:
  """Interprets expr using env and module's top level bindings.

  Args:
    module: The module that this expression is inside of.
    type_info: Mapping from AST node to its deduced/checked type.
    typecheck: Callback that can be used to get type info for a module.
    import_cache: Caches imported module info.
    expr: Expression to evaluate using the values from env and top level
      bindings (e.g. constants, other functions).
    env: Mapping from symbols to their integer values.
    bit_widths: Mapping from symbols to their bitwidths.
    fn_ctx: The (module name, function name, symbolic bindings) we are currently
      using.

  Returns:
    The integer value of the interpreted expression.

  Raises:
    KeyError: Occurs when the interpreter encounters a symbol that isn't in env.
  """
  return interpreter.Interpreter.interpret_expr(module, type_info, typecheck,
                                                import_cache, env, bit_widths,
                                                expr, FnCtx(*fn_ctx))

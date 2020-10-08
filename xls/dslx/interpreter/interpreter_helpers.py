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

from typing import Tuple, Optional, Dict
from xls.dslx import deduce
from xls.dslx.interpreter import interpreter
from xls.dslx.interpreter.bindings import FnCtx
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_type_info as type_info_mod

SymbolicBindings = Tuple[Tuple[str, int], ...]


def interpret_expr(module: ast.Module,
                   type_info: type_info_mod.TypeInfo,
                   env: Dict[str, int],
                   bit_widths: Dict[str, int],
                   expr: ast.Expr,
                   f_import: Optional[deduce.ImportFn],
                   fn_ctx=Tuple[str, str, SymbolicBindings]) -> int:
  """Interprets expr using env and module's top level bindings.

  Args:
    module: The module that this expression is inside of.
    type_info: Mapping from AST node to its deduced/checked type.
    env: Mapping from symbols to their integer values.
    bit_widths: Mapping from symbols to their bitwidths.
    expr: Expression to evaluate using the values from env and top level
      bindings (e.g. constants, other functions).
    f_import: Import routine callback for the interpreter to use.
    fn_ctx: The (module name, function name, symbolic bindings) we are currently
      using.

  Returns:
    The integer value of the interpreted expression.

  Raises:
    KeyError: Occurs when the interpreter encounters a symbol that isn't in env.
  """
  interp = interpreter.Interpreter(module, type_info, f_import=f_import)
  bindings = interp._make_top_level_bindings(module)  # pylint: disable=protected-access
  bindings.fn_ctx = FnCtx(*fn_ctx)
  for ident, val in env.items():
    bindings.add_value(
        ident,
        interpreter.Value.make_ubits(value=val, bit_count=bit_widths[ident]))
  return interp.evaluate_expr(expr, bindings).bits_payload.value

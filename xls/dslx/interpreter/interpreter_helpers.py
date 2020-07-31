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
"""Helper utilities for use with the interpreter.Interpreter."""

from typing import Tuple, Text, Optional, Dict
from xls.dslx import ast
from xls.dslx import deduce
from xls.dslx.interpreter import interpreter

SymbolicBindings = Tuple[Tuple[Text, int], ...]


def interpret_expr(module: ast.Module,
                   node_to_type: deduce.NodeToType,
                   env: Dict[Text, int],
                   bit_widths: Dict[Text, int],
                   expr: ast.Expr,
                   f_import: Optional[deduce.ImportFn],
                   fn_ctx=Tuple[Text, Text, SymbolicBindings]) -> int:
  """Interprets expr using env and module's top level bindings.
  Args:
    module: The module that this expression is inside of.
    node_to_type: Mapping from AST node to its deduced/checked type.
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
  interp = interpreter.Interpreter(module, node_to_type, f_import=f_import)
  bindings = interp._make_top_level_bindings(module)  # pylint: disable=protected-access
  bindings.fn_ctx = fn_ctx
  for ident, val in env.items():
    bindings.add_value(
        ident,
        interpreter.Value.make_ubits(value=val, bit_count=bit_widths[ident]))
  return interp.evaluate_expr(expr, bindings).bits_payload

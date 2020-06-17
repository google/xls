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
from typing import Sequence, Tuple
from xls.dslx import ast
from xls.dslx import deduce
from xls.dslx.interpreter import interpreter

def interpret_expr(module: ast.Module, node_to_type: deduce.NodeToType,
                   env: Sequence[Tuple[str, interpreter.Value]],
                   expr: ast.Expr) -> int:
  """Creates an Interpreter on-the-fly to evaluate expr with env"""
  interp = interpreter.Interpreter(module, node_to_type, f_import=None)
  bindings = interpreter.Bindings()
  for ident, val in env.items():
    bindings.add_value(ident, interpreter.Value.make_ubits(value=val,
                                                           bit_count=32))
  return interp.evaluate_expr(expr, bindings).bits_payload.value


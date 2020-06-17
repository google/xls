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

from xls.dslx.interpreter import interpreter

class SimpleInterpreter(object):
  """Wrapper class for Interpreter that exposes basic functionality.

  Attributes:
    interp: Interpreter object
    bindings: Bindings object to use with the interpreter (e.g. in expr evaluation)
  """

  def __init__(self, module, node_to_type):
    self.interp = interpreter.Interpreter(module, node_to_type, f_import=None)
    self.bindings = interpreter.Bindings()

  def add_binding(self, ident, val):
    """Adds {ident: (val as u32)} to this instance's Bindings"""
    self.bindings.add_value(ident, interpreter.Value.make_ubits(value=val,
                                                                bit_count=32))

  def evaluate(self, expr) -> int:
    """Evaluates expr using current Bindings and returns resulting bit value"""
    return self.interp.evaluate_expr(expr, self.bindings).bits_payload.value

def make_interpreter(mod, node_to_type) -> SimpleInterpreter:
  """Return a new SimpleInterpreter. To be used as a callback function."""
  return SimpleInterpreter(mod, node_to_type)

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
"""Helpers for running a quickcheck node given an interpreter."""

from xls.dslx import ir_name_mangler
from xls.dslx.interpreter import jit_comparison
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_concrete_type as concrete_type
from xls.dslx.python import interpreter
from xls.ir.python import package
from xls.jit.python import ir_jit


def run_quickcheck(interp: interpreter.Interpreter, ir_package: package.Package,
                   quickcheck: ast.QuickCheck, seed: int) -> None:
  """Runs a quickcheck AST node (via the LLVM JIT)."""
  fn = quickcheck.f
  ir_name = ir_name_mangler.mangle_dslx_name(fn.name.identifier,
                                             fn.get_free_parametric_keys(),
                                             interp.module, ())

  ir_function = ir_package.get_function(ir_name)
  argsets, results = ir_jit.quickcheck_jit(ir_function, seed,
                                           quickcheck.test_count)
  last_result = results[-1].get_bits().to_uint()
  if not last_result:
    last_argset = argsets[-1]
    fn_type = interp.type_info.get_type(fn)
    assert isinstance(fn_type, concrete_type.FunctionType), fn_type
    fn_param_types = fn_type.params
    dslx_argset = [
        str(jit_comparison.ir_value_to_interpreter_value(arg, arg_type))
        for arg, arg_type in zip(last_argset, fn_param_types)
    ]
    interpreter.throw_fail_error(
        fn.span, f'Found falsifying example after '
        f'{len(results)} tests: {dslx_argset}')

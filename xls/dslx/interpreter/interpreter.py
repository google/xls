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

"""Interpreter for the AST data structure.

Used as a reference for evaluating modules to a value from its syntactic form.

This is a complement to other execution modes that can help sanity check more
optimized forms of execution.
"""

import contextlib
import functools
import sys
from typing import Text, Optional, Tuple, Callable, Sequence

from absl import logging
import termcolor

from xls.dslx import ir_name_mangler
from xls.dslx.interpreter import jit_comparison
from xls.dslx.interpreter.errors import EvaluateError
from xls.dslx.parametric_instantiator import SymbolicBindings
from xls.dslx.python import builtins
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_evaluate
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import FunctionType
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_pos import Span
from xls.dslx.python.interp_bindings import Bindings
from xls.dslx.python.interp_bindings import FnCtx
from xls.dslx.python.interp_value import Builtin
from xls.dslx.python.interp_value import Value
from xls.ir.python import package as ir_package_mod
from xls.jit.python import ir_jit


class _WipSentinel:
  """Marker to show that something is the in process of being evaluated."""


ImportSubject = Tuple[Text, ...]
ImportInfo = Tuple[ast.Module, type_info_mod.TypeInfo]


class Interpreter:
  """Object that interprets an AST of expressions to evaluate it to a value."""

  def __init__(self,
               module: ast.Module,
               type_info: type_info_mod.TypeInfo,
               f_import: Optional[Callable[[ImportSubject], ImportInfo]],
               trace_all: bool = False,
               ir_package: Optional[ir_package_mod.Package] = None):
    self._module = module
    self._type_info = type_info
    self._wip = {}  # Work-in-progress constant evaluation annotations.
    self._f_import = f_import
    self._trace_all = trace_all
    self._ir_package = ir_package

  # Functions in the interpreter conform to shared abstract signatures.
  # pylint: disable=unused-argument

  def _evaluate_ConstantArray(  # pylint: disable=invalid-name
      self, expr: ast.ConstantArray, bindings: Bindings,
      type_context: Optional[ConcreteType]) -> Value:
    """Evaluates a 'ConstantArray' AST node to a value."""
    return cpp_evaluate.evaluate_Array(expr, bindings, type_context,
                                       self._get_callbacks())

  def _call_builtin_fn(
      self,
      builtin: Builtin,
      args: Sequence[Value],
      span: Span,
      invocation: Optional[ast.Invocation] = None,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Calls a builtin function identified via a 'builtin' enum value."""
    name = builtin.to_name()
    if hasattr(builtins, name):
      f = getattr(builtins, name)
    else:
      f = getattr(self, f'_builtin_{name}')
    result = f(args, span, invocation, symbolic_bindings)
    assert isinstance(result, Value), (result, f)
    return result

  def _call_fn_value(
      self,
      fv: Value,
      args: Sequence[Value],
      span: Span,
      invocation: Optional[ast.Invocation] = None,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Calls function values, either a builtin or a user defined function."""
    if fv.is_builtin_function():
      return self._call_builtin_fn(fv.get_builtin_fn(), args, span, invocation,
                                   symbolic_bindings)
    else:
      _, f = fv.get_user_fn_data()

    return self._evaluate_fn(f, args, span, invocation, symbolic_bindings)

  def _evaluate_Invocation(  # pylint: disable=invalid-name
      self, expr: ast.Invocation, bindings: Bindings,
      _: Optional[ConcreteType]) -> Optional[Value]:
    """Evaluates an 'Invocation' AST node to a value."""
    if self._trace_all and isinstance(
        expr.callee,
        ast.NameRef) and expr.callee.name_def.identifier == 'trace':
      # Safe to skip this and return nothing if this is a trace invocation;
      # trace isn't an input to any downstream expressions.
      return Value.make_nil()
    arg_values = [self._evaluate(arg, bindings) for arg in expr.args]
    callee_value = self._evaluate(expr.callee, bindings)
    if not callee_value.is_function():
      raise EvaluateError(
          expr.callee.span,
          'Callee value is not a function (should have been determined during type inference); got: {}'
          .format(callee_value))
    fn_symbolic_bindings = ()
    if bindings.fn_ctx:
      # The symbolic bindings of this invocation were already computed during
      # typechecking.
      fn_symbolic_bindings = self._type_info.get_invocation_symbolic_bindings(
          expr, bindings.fn_ctx.sym_bindings)
    return self._call_fn_value(
        callee_value,
        arg_values,
        expr.span,
        expr,
        symbolic_bindings=fn_symbolic_bindings)

  def _perform_trace(self, lhs: Text, span: Span, value: Value) -> None:
    """Actually writes the tracing output to stderr."""
    leader = 'trace of {} @ {}:'.format(lhs, span)
    if sys.stderr.isatty():
      print(termcolor.colored(leader, color='blue'), value, file=sys.stderr)
    else:
      print(leader, value, file=sys.stderr)

  def _optional_trace(self, expr: ast.Expr, result: Value) -> None:
    """Traces the current experession if "trace all" mode is active.

    Args:
      expr: The expression to trace.
      result: The result of evaluating the given expression.
    """
    # We don't need to trace trace (obv), or Lets - we just want to see the
    # non-Let bodies.
    # NameRefs and ModRefs also add a lot of noise w/o a lot of value.
    is_trace_instance = isinstance(expr, ast.Invocation) and isinstance(
        expr.callee, ast.NameRef) and expr.callee.name_def.identifier == 'trace'
    is_let_instance = isinstance(expr, ast.Let)

    if (not is_trace_instance and not is_let_instance and
        not result.is_function()):
      self._perform_trace(str(expr), expr.span, result)

  def _evaluate(self,
                expr: ast.Expr,
                bindings: Bindings,
                type_context: Optional[ConcreteType] = None) -> Value:
    """Entry point for evaluating an expression to a value.

    Args:
      expr: Expression AST node to evaluate.
      bindings: Current bindings for this evaluation (i.e. ident: value map).
      type_context: If a type is deduced from surrounding context, it is
        provided via this argument.

    Returns:
      The value that the AST node evaluates to.

    Raises:
      EvaluateError: If an error occurs during evaluation. This also attempts to
        print a rough expression-stack-trace for determining the provenance of
        an error to stderr.
    """
    clsname = expr.__class__.__name__
    logging.vlog(3, 'Evaluating %s: %s', clsname, expr)
    handler = getattr(cpp_evaluate, f'evaluate_{clsname}', None)
    if handler is None:
      handler = getattr(self, '_evaluate_{}'.format(clsname))
    else:  # We give the callback data to the C++ handlers.
      handler = functools.partial(handler, callbacks=self._get_callbacks())

    try:
      result = handler(expr, bindings, type_context)
      logging.vlog(3, 'Evaluated %s: %s => %s', clsname, expr, result)
      if self._trace_all and result is not None:
        self._optional_trace(expr, result)
      assert isinstance(result, Value), (result, handler)
      return result
    except (AssertionError, EvaluateError, TypeError) as e:
      # Give some more helpful traceback context in expression evaluation for
      # where errors come from.
      if isinstance(e, AssertionError):
        kind = 'assertion'
      elif isinstance(e, TypeError):
        kind = 'python type'
      else:
        kind = 'evaluation'
      print('{} error @ {}: {}'.format(kind, expr.span, e), file=sys.stderr)
      raise

  def evaluate_literal(self, expr: ast.Expr) -> Value:
    return self._evaluate(expr, Bindings(None))

  def evaluate_expr(self, expr: ast.Expr, bindings: Bindings) -> Value:
    """Evaluates a stand-alone expression with the given bindings."""
    return self._evaluate(expr, bindings)

  def _builtin_map(
      self,
      args: Sequence[Value],
      span: Span,
      expr: ast.Invocation,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Implements the 'map' builtin."""
    if len(args) != 2:
      raise EvaluateError(
          span,
          'Invalid number of arguments to map; got {} want 2'.format(len(args)))
    inputs, map_fn = args
    outputs = []
    for input_ in inputs.get_elements():
      ret = self._call_fn_value(map_fn, [input_], span, expr, symbolic_bindings)
      outputs.append(ret)

    return Value.make_array(tuple(outputs))

  def _builtin_trace(
      self,
      args: Sequence[Value],
      span: Span,
      expr: ast.Invocation,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Implements the 'trace' builtin."""
    if len(args) != 1:
      raise ValueError(
          'Invalid number of arguments to trace; got {} want 1'.format(
              len(args)))

    self._perform_trace(expr.format_args(), span, args[0])
    assert isinstance(args[0], Value), args[0]
    return args[0]

  def _evaluate_fn(
      self,
      fn: ast.Function,
      args: Sequence[Value],
      span: Span,
      expr: Optional[ast.Invocation] = None,
      symbolic_bindings: Optional[SymbolicBindings] = None) -> Value:
    """Wraps _eval_fn_with_interpreter() to compare with JIT execution.

    Unless this Interpreter was created with an ir_package, this does nothing
    more than call _eval_fn_with_interpreter(). Otherwise, fn is executed with
    the LLVM IR JIT and its return value is compared against the interpreted
    value as a consistency check.

    TODO(hjmontero): 2020-8-4 This OK because there are no side effects. We
    should investigate what happens when there are side effects (e.g. DSLX fatal
    errors).

    Args:
      fn: Function to evaluate.
      args: Actual arguments used to invoke the function.
      span: Span of the invocation causing this evaluation.
      expr: Invocation AST node causing this evaluation.
      symbolic_bindings: Symbolic bindings to be used for this function
        evaluation present (if the function is parameteric).

    Returns:
      The value that results from DSL interpretation.
    """
    has_child_type_info = expr and self._type_info.has_instantiation(
        expr, symbolic_bindings)
    invocation_type_info = (
        self._type_info.get_instantiation(expr, symbolic_bindings)
        if has_child_type_info else self._type_info)

    @contextlib.contextmanager
    def ntt_swap(new_ntt):
      old_ntt = self._type_info
      self._type_info = new_ntt
      yield
      self._type_info = old_ntt

    with ntt_swap(invocation_type_info):
      interpreter_value = cpp_evaluate.evaluate_function(
          fn, args, span, symbolic_bindings, self._get_callbacks())

    ir_name = ir_name_mangler.mangle_dslx_name(fn.name.identifier,
                                               fn.get_free_parametric_keys(),
                                               fn.get_containing_module(),
                                               symbolic_bindings)

    if self._ir_package:
      # TODO(hjmontero): 2020-07-28 Cache JIT function so we don't have to
      # create it every time. This requires us to figure out how to wrap
      # IrJit::Create().
      ir_function = self._ir_package.get_function(ir_name)
      try:
        ir_args = jit_comparison.convert_args_to_ir(args)

        jit_value = ir_jit.ir_jit_run(ir_function, ir_args)
        jit_comparison.compare_values(interpreter_value, jit_value)
      except (jit_comparison.UnsupportedJitConversionError,
              jit_comparison.JitMiscompareError) as e:
        builtins.throw_fail_error(expr.span if expr else fn.span, str(e))

    return interpreter_value

  def _get_callbacks(self) -> cpp_evaluate.InterpCallbackData:
    """Returns a set of callbacks that cpp_evaluate can use.

    This allows cpp_evaluate to request assist from the (Python) interpreter
    before things have been fully ported over.
    """

    def is_wip(c: ast.Constant) -> Optional[Value]:
      """Returns whether the constant is in the process of being computed."""
      status = self._wip.get(c)
      logging.vlog(3, 'Constant eval status %r: %r', c, status)
      return status is _WipSentinel

    def note_wip(c: ast.Constant, v: Optional[Value]) -> Optional[Value]:
      assert isinstance(c, ast.Constant), repr(c)
      assert v is None or isinstance(v, Value), repr(v)

      if v is None:  # Starting evaluation, attempting to mark as WIP.
        current = self._wip.get(c)
        if current is not None and current is not _WipSentinel:
          assert isinstance(current, Value), repr(current)
          return current  # Already computed.
        logging.vlog(3, 'Noting WIP constant eval: %r', c)
        self._wip[c] = _WipSentinel
        return None

      logging.vlog(3, 'Noting complete constant eval: %r => %r', c, v)
      self._wip[c] = v
      return v

    # Hack to avoid circular dependency on the importer definition.
    typecheck = getattr(self._f_import, '_typecheck', None)
    cache = getattr(self._f_import, '_cache', None)

    def get_type_info() -> type_info_mod.TypeInfo:
      return self._type_info

    return cpp_evaluate.InterpCallbackData(typecheck, self._evaluate, is_wip,
                                           note_wip, get_type_info, cache)

  def _make_top_level_bindings(self, m: ast.Module) -> Bindings:
    """Creates a fresh set of bindings for use in module-level evaluation.

    Args:
      m: The module that the top level bindings are being made for, used to
        populate constants / enums.

    Returns:
      Bindings containing builtins and function identifiers at the top level of
      the module.
    """
    result = cpp_evaluate.make_top_level_bindings(m, self._get_callbacks())
    assert '__top_level_bindings_' + m.name in result.keys(), (m, result.keys())
    return result

  def run_quickcheck(self, quickcheck: ast.QuickCheck, seed: int) -> None:
    """Runs a quickcheck AST node (via the LLVM JIT)."""
    assert self._ir_package
    fn = quickcheck.f
    ir_name = ir_name_mangler.mangle_dslx_name(fn.name.identifier,
                                               fn.get_free_parametric_keys(),
                                               self._module, ())

    ir_function = self._ir_package.get_function(ir_name)
    argsets, results = ir_jit.quickcheck_jit(ir_function, seed,
                                             quickcheck.test_count)
    last_result = results[-1].get_bits().to_uint()
    if not last_result:
      last_argset = argsets[-1]
      fn_type = self._type_info[fn]
      assert isinstance(fn_type, FunctionType), fn_type
      fn_param_types = fn_type.params
      dslx_argset = [
          str(jit_comparison.ir_value_to_interpreter_value(arg, arg_type))
          for arg, arg_type in zip(last_argset, fn_param_types)
      ]
      builtins.throw_fail_error(
          fn.span, f'Found falsifying example after '
          f'{len(results)} tests: {dslx_argset}')

  def run_test(self, name: Text) -> None:
    bindings = self._make_top_level_bindings(self._module)
    test = self._module.get_test(name)
    assert test.name.identifier == name
    logging.vlog(1, 'Running test: %s', test)
    bindings.fn_ctx = FnCtx(self._module.name, '{}_test'.format(name), ())
    result = self._evaluate(test.body, bindings)
    if not result.is_nil_tuple():
      raise EvaluateError(
          test.body.span,
          'Want test to return nil tuple, but got {}'.format(result))

  def run_function(self, name: Text, args: Sequence[Value]) -> Value:
    f = self._module.get_function(name)
    assert not f.is_parametric()
    fake_pos = Pos('<fake>', 0, 0)
    fake_span = Span(fake_pos, fake_pos)
    return self._evaluate_fn(f, args, fake_span, symbolic_bindings=())

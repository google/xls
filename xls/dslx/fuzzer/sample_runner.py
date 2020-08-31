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

"""Library for operating on a generated code sample in the fuzzer."""

import collections
import os
import subprocess
import time
from typing import Text, Tuple, Optional, Dict, Sequence

from absl import logging

from xls.common import check_simulator
from xls.common import revision
from xls.common import runfiles
from xls.common.xls_error import XlsError
from xls.dslx import ast
from xls.dslx import concrete_type as concrete_type_mod
from xls.dslx import parse_and_typecheck
from xls.dslx import type_info as type_info_mod
from xls.dslx import typecheck
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.fuzzer import sample
from xls.dslx.interpreter.interpreter import Interpreter
from xls.dslx.interpreter.value import Value
from xls.dslx.interpreter.value_parser import value_from_string
from xls.ir.python import ir_parser
from xls.ir.python import value as ir_value_mod
from xls.ir.python.format_preference import FormatPreference

IR_CONVERTER_MAIN_PATH = runfiles.get_path('xls/dslx/ir_converter_main')
EVAL_IR_MAIN_PATH = runfiles.get_path('xls/tools/eval_ir_main')
IR_OPT_MAIN_PATH = runfiles.get_path('xls/tools/opt_main')
CODEGEN_MAIN_PATH = runfiles.get_path('xls/tools/codegen_main')
SIMULATE_MODULE_MAIN_PATH = runfiles.get_path('xls/tools/simulate_module_main')


class SampleError(XlsError):
  """Error raised if any problem is encountered running the sample.

  These issues can include parsing errors, tools crashing, or result
  miscompares(among others).
  """
  pass


def ir_value_to_interpreter_value(value: ir_value_mod.Value) -> Value:
  """Converts an IR Value to an interpreter Value."""
  if value.is_bits():
    if value.get_bits().bit_count() <= 64:
      return Value.make_ubits(value.get_bits().bit_count(),
                              value.get_bits().to_uint())
    else:
      # For wide values which do not fit in 64 bits, parse value as as string.
      return value_from_string(value.to_str(FormatPreference.HEX))

  elif value.is_array():
    return Value.make_array(
        tuple(ir_value_to_interpreter_value(e) for e in value.get_elements()))
  else:
    assert value.is_tuple()
    return Value.make_tuple(
        tuple(ir_value_to_interpreter_value(e) for e in value.get_elements()))


def sign_convert_value(concrete_type: ConcreteType, value: Value) -> Value:
  """Converts the values to matched the signedness of the concrete type.

  Converts bits-typed Values contained within the given Value to match the
  signedness of the ConcreteType. Examples:

  invocation: sign_convert_value(s8, u8:64)
  returns: s8:64

  invocation: sign_convert_value(s3, u8:7)
  returns: s3:-1

  invocation: sign_convert_value((s8, u8), (u8:42, u8:10))
  returns: (s8:42, u8:10)

  This conversion functionality is required because the Values used in the DSLX
  may be signed while Values in IR interpretation and Verilog simulation are
  always unsigned.

  This function is idempotent.

  Args:
    concrete_type: ConcreteType to match.
    value: Input value.

  Returns:
    Sign-converted value.
  """
  if isinstance(concrete_type, concrete_type_mod.TupleType):
    assert value.is_tuple()
    assert len(value.tuple_members) == concrete_type.get_tuple_length()
    return Value.make_tuple(
        tuple(
            sign_convert_value(t, a) for t, a in zip(
                concrete_type.get_unnamed_members(), value.tuple_members)))
  elif isinstance(concrete_type, concrete_type_mod.ArrayType):
    assert value.is_array()
    assert len(value.array_payload.elements) == concrete_type.size
    return Value.make_array(
        tuple(
            sign_convert_value(concrete_type.get_element_type(), v)
            for v in value.array_payload.elements))
  elif concrete_type_mod.is_sbits(concrete_type):
    return Value.make_sbits(value.get_bit_count(), value.get_bits_value())
  else:
    assert concrete_type_mod.is_ubits(concrete_type)
    return value


def sign_convert_args_batch(f: ast.Function, m: ast.Module,
                            args_batch: sample.ArgsBatch) -> sample.ArgsBatch:
  """Sign-converts ArgsBatch to match the signedness of function arguments."""
  f = m.get_function('main')
  type_info = typecheck.check_module(m, f_import=None)
  arg_types = tuple(type_info[p.type_] for p in f.params)
  converted_batch = []
  for args in args_batch:
    assert len(arg_types) == len(args)
    converted_batch.append(
        tuple(sign_convert_value(t, a) for t, a in zip(arg_types, args)))
  return tuple(converted_batch)


class SampleRunner:
  """A class for performing various operations on a code sample.

  Code sample can be in DSLX or IR. The possible operations include:
     * Converting DSLX to IR (DSLX input only).
     * Interpeting the code with supplied arguments.
     * Optimizing IR.
     * Generating Verilog.
     * Simulating Verilog.
     * Comparing interpreter/simulation results for equality.
  The runner operates in a single directory supplied at construction time and
  records all state, command invocations, and outputs to that directory to
  enable easier debugging and replay.
  """

  def __init__(self, run_dir: Text):
    self._run_dir = run_dir

  def run(self, smp: sample.Sample):
    """Runs the given sample.

    Args:
      smp: Sample to run.

    Raises:
      SampleError: If an error was encountered.
    """
    if smp.options.input_is_dslx:
      input_filename = self._write_file('sample.x', smp.input_text)
    else:
      input_filename = self._write_file('sample.ir', smp.input_text)
    json_options_filename = self._write_file('options.json',
                                             smp.options.to_json())
    args_filename = None
    if smp.args_batch is not None:
      args_filename = self._write_file(
          'args.txt', sample.args_batch_to_text(smp.args_batch))

    self.run_from_files(input_filename, json_options_filename, args_filename)

  def run_from_files(self, input_filename: Text, json_options_filename: Text,
                     args_filename: Text):
    """Runs a sample which is read from files.

    Each filename must be the name of a file (not a full path) which is
    contained in the SampleRunner's run directory.

    Args:
      input_filename: The filename of the sample code.
      json_options_filename: The filename of the JSON-serialized SampleOptions.
      args_filename: The optional filename of the serialized ArgsBatch.

    Raises:
      SampleError: If an error was encountered.
    """
    logging.vlog(1, 'Reading sample files.')
    input_text = self._read_file(input_filename)
    options = sample.SampleOptions.from_json(
        self._read_file(json_options_filename))
    args_batch: Optional[sample.ArgsBatch] = None
    if args_filename:
      args_batch = sample.parse_args_batch(self._read_file(args_filename))

    self._write_file('revision.txt', revision.get_revision())

    # Gather results in an OrderedDict because the first entered result is used
    # as a reference.
    results = collections.OrderedDict()  # type: Dict[Text, Sequence[Value]]

    try:
      logging.vlog(1, 'Parsing DSLX file.')
      start = time.time()
      if options.input_is_dslx:
        m, type_info = parse_and_typecheck.parse_text_fakefs(
            input_text,
            'test_module',
            f_import=None,
            print_on_error=True,
            filename='/fake/test_module.x')
        logging.vlog(1, 'Parsing DSLX file complete, elapsed %0.2fs',
                     time.time() - start)

        if args_batch is not None:
          logging.vlog(1, 'Interpreting DSLX file.')
          start = time.time()
          results['interpreted DSLX'] = self._interpret_dslx(
              m, type_info, args_batch)
          logging.vlog(1, 'Parsing DSLX file complete, elapsed %0.2fs',
                       time.time() - start)

        if not options.convert_to_ir:
          return
        ir_filename = self._dslx_to_ir(input_filename)
      else:
        ir_filename = self._write_file('sample.ir', input_text)

      if args_filename is not None:
        # Unconditionally evaluate with the interpreter even if using the
        # JIT. This exercises the interpreter and serves as a reference.
        results['evaluated unopt IR (interpreter)'] = self._evaluate_ir(
            ir_filename, args_filename, False)
        if options.use_jit:
          results['evaluated unopt IR (JIT)'] = self._evaluate_ir(
              ir_filename, args_filename, True)

      if options.optimize_ir:
        opt_ir_filename = self._optimize_ir(ir_filename)

        if args_filename is not None:
          if options.use_jit:
            results['evaluated opt IR (JIT)'] = self._evaluate_ir(
                opt_ir_filename, args_filename, True)
          else:
            results['evaluated opt IR (interpreter)'] = self._evaluate_ir(
                opt_ir_filename, args_filename, False)
        if options.codegen:
          verilog_filename = self._codegen(opt_ir_filename,
                                           options.codegen_args)
          if options.simulate:
            assert args_filename is not None
            results['simulated'] = self._simulate(verilog_filename,
                                                  'module_sig.textproto',
                                                  args_filename,
                                                  options.simulator)

      self._compare_results(results, args_batch)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Exception when running sample: %s', str(e))
      self._write_file('exception.txt', str(e))
      raise SampleError(str(e))

  def _run_command(self, desc: Text, args: Sequence[Text]):
    """Runs the given commands.

    Args:
      desc: Textual description of what the command is doing. Emitted to stdout.
      args: The command line arguments.

    Returns:
      Stdout of the command.

    Raises:
      subprocess.CalledProcessError: If subprocess returns non-zero code.
    """
    # Print the command line with the runfiles directory prefix elided to reduce
    # clutter.
    if logging.get_verbosity() > 0:
      args = list(args) + ['-v={}'.format(logging.get_verbosity())]
    cmd_line = subprocess.list2cmdline(args)
    logging.vlog(1, '%s:  %s', desc, cmd_line)
    start = time.time()
    basename = os.path.basename(args[0])
    stderr_path = os.path.join(self._run_dir, basename + '.stderr')
    with open(stderr_path, 'w') as f_stderr:
      comp = subprocess.run(
          list(args) + ['--logtostderr'],
          cwd=self._run_dir,
          stdout=subprocess.PIPE,
          stderr=f_stderr,
          check=False)

    if logging.vlog_is_on(4):
      logging.vlog(4, '{} stdout:'.format(basename))
      # stdout and stderr can be long so split them by line to avoid clipping.
      for line in comp.stdout.decode('utf-8').splitlines():
        logging.vlog(4, line)

      logging.vlog(4, '{} stderr:'.format(basename))
      with open(stderr_path, 'r') as f:
        for line in f.read().splitlines():
          logging.vlog(4, line)

    logging.vlog(1, '%s complete, elapsed %0.2fs', desc, time.time() - start)

    comp.check_returncode()

    return comp.stdout.decode('utf-8')

  def _write_file(self, filename: Text, content: Text) -> Text:
    """Writes the given content into a named file in the run directory."""
    with open(os.path.join(self._run_dir, filename), 'w') as f:
      f.write(content)
    return filename

  def _read_file(self, filename: Text) -> Text:
    """Returns the content of the named text file in the run directory."""
    with open(os.path.join(self._run_dir, filename), 'r') as f:
      return f.read()

  def _compare_results(self, results: Dict[Text, Sequence[Value]],
                       args_batch: Optional[sample.ArgsBatch]):
    """Compares a set of results as for equality.

    Each entry in the map is sequence of Values generated from some source
    (e.g., interpreting the optimized IR). Each sequence of Values is compared
    for equality.

    Args:
      results: Map of result Values.
      args_batch: Optional batch of arguments used to produce the given results.
        Batch should be the same length as the number of results for any given
        value in "results".

    Raises:
      SampleError: A miscompare is found.
    """
    if not results:
      return

    if args_batch:  # Check length is the same as results.
      assert len(next(iter(results.values()))) == len(args_batch)

    reference = None
    for name, values in results.items():
      if reference is None:
        reference = name
      else:
        if len(results[reference]) != len(values):
          raise SampleError(
              f'Results for {reference} has {len(results[reference])} values, {name} has {len(values)}'
          )
        for i in range(len(values)):
          ref_result = results[reference][i]
          # The IR tools and the verilog simulator produce unsigned values while
          # the DSLX interpreter can produce signed values so compare the
          # results ignoring signedness.
          if not ref_result.eq_ignore_sign(values[i]).is_true():
            args = '(args unknown)'
            if args_batch:
              args = '; '.join(str(a) for a in args_batch[i])
            raise SampleError(f'Result miscompare for sample {i}:'
                              f'\nargs: {args}'
                              f'\n{reference:40} = {ref_result}'
                              f'\n{name:40} = {values[i]}')

  def _interpret_dslx(self, m: ast.Module, type_info: type_info_mod.TypeInfo,
                      args_batch: sample.ArgsBatch) -> Tuple[Value, ...]:
    """Interprets the DSLX module returns the result Values."""
    interp = Interpreter(m, type_info, f_import=None)
    dslx_results = []
    f = m.get_function('main')
    for args in sign_convert_args_batch(f, m, args_batch):
      dslx_results.append(interp.run_function('main', args))
    self._write_file('sample.x.results',
                     '\n'.join(str(r) for r in dslx_results))
    return tuple(dslx_results)

  def _parse_values(self, s: Text) -> Tuple[Value, ...]:
    """Parses a line-deliminated sequence of text-formatted Values.

    Example of expected input:
      bits[32]:0x42
      bits[32]:0x123

    Args:
      s: Input string.

    Returns:
      Tuple of parsed Values.
    """

    def str_to_value(s: Text) -> Value:
      return ir_value_to_interpreter_value(
          ir_parser.Parser.parse_typed_value(s))

    return tuple(
        str_to_value(line.strip()) for line in s.split('\n') if line.strip())

  def _evaluate_ir(self, ir_filename: Text, args_filename: Text,
                   use_jit: bool) -> Tuple[Value, ...]:
    """Evaluate the IR file and returns the result Values."""
    results_text = self._run_command(
        'Evaluating IR file ({}): {}'.format(
            'JIT' if use_jit else 'interpreter', ir_filename),
        (EVAL_IR_MAIN_PATH, '--input_file=' + args_filename,
         '--use_llvm_jit' if use_jit else '--nouse_llvm_jit', ir_filename))
    self._write_file(ir_filename + '.results', results_text)
    return self._parse_values(results_text)

  def _dslx_to_ir(self, dslx_filename: Text) -> Text:
    """Converts the DSLX file to an IR file whose filename is returned."""
    ir_text = self._run_command('Converting DSLX to IR',
                                (IR_CONVERTER_MAIN_PATH, dslx_filename))
    return self._write_file('sample.ir', ir_text)

  def _optimize_ir(self, ir_filename: Text) -> Text:
    """Optimizes the IR file and returns the resulting filename."""
    opt_ir_text = self._run_command('Optimizing IR',
                                    (IR_OPT_MAIN_PATH, ir_filename))
    return self._write_file('sample.opt.ir', opt_ir_text)

  def _codegen(self, ir_filename: Text, codegen_args: Sequence[Text]) -> Text:
    """Generates Verilog from the IR file and return the Verilog filename."""
    args = [
        CODEGEN_MAIN_PATH, '--output_signature_path=module_sig.textproto',
        '--delay_model=unit'
    ]
    args.extend(codegen_args)
    args.append(ir_filename)
    verilog_text = self._run_command('Generating Verilog', args)
    return self._write_file('sample.v', verilog_text)

  def _simulate(self,
                verilog_filename: Text,
                module_sig_filename: Text,
                args_filename: Text,
                simulator: Optional[Text] = None) -> Tuple[Value, ...]:
    """Simulates the Verilog file and returns the results Values."""
    simulator_args = [
        SIMULATE_MODULE_MAIN_PATH,
        '--signature_file=' + module_sig_filename,
        '--args_file=' + args_filename
    ]
    if simulator:
      simulator_args.append('--verilog_simulator=' + simulator)
    simulator_args.append(verilog_filename)

    check_simulator.check_simulator(simulator)

    results_text = self._run_command(f'Simulating Verilog {verilog_filename}',
                                     simulator_args)
    self._write_file(verilog_filename + '.results', results_text)
    return self._parse_values(results_text)

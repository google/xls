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

"""Library for operating on a generated code sample in the fuzzer."""

import collections
import os
import subprocess
import time
from typing import Text, Tuple, Optional, Dict, Sequence, List

from absl import logging

from xls.common import check_simulator
from xls.common import revision
from xls.common import runfiles
from xls.common.xls_error import XlsError
from xls.dslx.python import interpreter
from xls.dslx.python.interp_value import Value
from xls.fuzzer import sample_summary_pb2
from xls.fuzzer.python import cpp_sample as sample

IR_CONVERTER_MAIN_PATH = runfiles.get_path('xls/dslx/ir_converter_main')
EVAL_IR_MAIN_PATH = runfiles.get_path('xls/tools/eval_ir_main')
IR_OPT_MAIN_PATH = runfiles.get_path('xls/tools/opt_main')
CODEGEN_MAIN_PATH = runfiles.get_path('xls/tools/codegen_main')
SIMULATE_MODULE_MAIN_PATH = runfiles.get_path('xls/tools/simulate_module_main')


ArgsBatch = List[List[Value]]


class SampleError(XlsError):
  """Error raised if any problem is encountered running the sample.

  These issues can include parsing errors, tools crashing, or result
  miscompares(among others).
  """
  pass


class Timer:
  """Timer for measuring the elapsed time of an operation in nanoseconds.

  Example usage:
    with Timer() as t:
      foo_bar()
    print('foobar() took ' + str(t.elapsed_ns) + 'us')
  """

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, *args):
    self.end = time.time()
    self.elapsed_ns = int((self.end - self.start) * 1e9)


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
    self.timing = sample_summary_pb2.SampleTimingProto()

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
    if smp.args_batch:
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
    args_batch: Optional[ArgsBatch] = None
    if args_filename:
      args_batch = sample.parse_args_batch(self._read_file(args_filename))

    self._write_file('revision.txt', revision.get_revision())

    # Gather results in an OrderedDict because the first entered result is used
    # as a reference.
    results = collections.OrderedDict()  # type: Dict[Text, Sequence[Value]]

    try:
      if options.input_is_dslx:
        if args_batch:
          logging.vlog(1, 'Interpreting DSLX file.')
          with Timer() as t:
            results['interpreted DSLX'] = self._interpret_dslx(
                input_text, 'main', args_batch)
          logging.vlog(1, 'Interpreting DSLX complete, elapsed %0.2fs',
                       t.elapsed_ns / 1e9)
          self.timing.interpret_dslx_ns = t.elapsed_ns

        if not options.convert_to_ir:
          return

        with Timer() as t:
          ir_filename = self._dslx_to_ir(input_filename)
        self.timing.convert_ir_ns = t.elapsed_ns
      else:
        ir_filename = self._write_file('sample.ir', input_text)

      if args_filename is not None:
        # Unconditionally evaluate with the interpreter even if using the
        # JIT. This exercises the interpreter and serves as a reference.
        with Timer() as t:
          results['evaluated unopt IR (interpreter)'] = self._evaluate_ir(
              ir_filename, args_filename, False)
        self.timing.unoptimized_interpret_ir_ns = t.elapsed_ns

        if options.use_jit:
          with Timer() as t:
            results['evaluated unopt IR (JIT)'] = self._evaluate_ir(
                ir_filename, args_filename, True)
          self.timing.unoptimized_jit_ns = t.elapsed_ns

      if options.optimize_ir:
        with Timer() as t:
          opt_ir_filename = self._optimize_ir(ir_filename)
        self.timing.optimize_ns = t.elapsed_ns

        if args_filename is not None:
          if options.use_jit:
            with Timer() as t:
              results['evaluated opt IR (JIT)'] = self._evaluate_ir(
                  opt_ir_filename, args_filename, True)
            self.timing.optimized_jit_ns = t.elapsed_ns
          with Timer() as t:
            results['evaluated opt IR (interpreter)'] = self._evaluate_ir(
                opt_ir_filename, args_filename, False)
          self.timing.optimized_interpret_ir_ns = t.elapsed_ns

        if options.codegen:
          with Timer() as t:
            verilog_filename = self._codegen(opt_ir_filename,
                                             options.codegen_args)
          self.timing.codegen_ns = t.elapsed_ns

          if options.simulate:
            assert args_filename is not None
            with Timer() as t:
              results['simulated'] = self._simulate(verilog_filename,
                                                    'module_sig.textproto',
                                                    args_filename,
                                                    options.simulator)
            self.timing.simulate_ns = t.elapsed_ns

      self._compare_results(results, args_batch)
    except Exception as e:  # pylint: disable=broad-except
      # Note: this is a bit of a hack because pybind11 doesn't make it very
      # possible to define custom __str__ on exception types. Our C++ exception
      # types have a field called "message" generally so we look for that.
      msg = e.message if hasattr(e, 'message') else str(e)
      logging.exception('Exception when running sample: %s', msg)
      self._write_file('exception.txt', msg)
      raise SampleError(msg)

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
                       args_batch: Optional[ArgsBatch]):
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

    # Returns whether the two given values are equal. The IR tools and the
    # verilog simulator produce unsigned values while the DSLX interpreter can
    # produce signed values so compare the results ignoring signedness.
    def values_equal(a: Value, b: Value) -> bool:
      return a.eq(b).is_true()

    reference = None
    for name in sorted(results.keys()):
      values = results[name]
      if reference is None:
        reference = name
      else:
        if len(results[reference]) != len(values):
          raise SampleError(
              f'Results for {reference} has {len(results[reference])} values,'
              f' {name} has {len(values)}')

        for i in range(len(values)):
          ref_result = results[reference][i]
          if not values_equal(ref_result, values[i]):
            # Bin all of the sources by whether they match the reference or
            # 'values'. This helps identify which of the two is likely
            # correct.
            reference_matches = sorted(
                n for n, v in results.items() if values_equal(v[i], ref_result))
            values_matches = sorted(
                n for n, v in results.items() if values_equal(v[i], values[i]))
            args = '(args unknown)'
            if args_batch:
              args = '; '.join(a.to_ir_str() for a in args_batch[i])
            raise SampleError(f'Result miscompare for sample {i}:'
                              f'\nargs: {args}'
                              f'\n{", ".join(reference_matches)} ='
                              f'\n   {ref_result.to_ir_str()}'
                              f'\n{", ".join(values_matches)} ='
                              f'\n   {values[i].to_ir_str()}')

  def _interpret_dslx(self, text: str, function_name: str,
                      args_batch: ArgsBatch) -> Tuple[Value, ...]:
    """Interprets the DSLX module returns the result Values."""
    dslx_results = interpreter.run_batched(text, function_name, args_batch)
    self._write_file('sample.x.results',
                     '\n'.join(r.to_ir_str() for r in dslx_results))
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

    return tuple(
        interpreter.ir_value_text_to_interp_value(line.strip())
        for line in s.split('\n')
        if line.strip())

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

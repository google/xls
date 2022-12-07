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

import os
import subprocess
import time
from typing import Tuple, Optional, Dict, Sequence, List

from absl import logging

from xls.common import check_simulator
from xls.common import revision
from xls.common import runfiles
from xls.common.xls_error import XlsError
from xls.dslx.python import create_import_data
from xls.dslx.python import interpreter
from xls.dslx.python import parse_and_typecheck
from xls.dslx.python.interp_value import interp_value_from_ir_string
from xls.dslx.python.interp_value import Value
from xls.fuzzer import sample_summary_pb2
from xls.fuzzer.python import cpp_sample as sample
from xls.ir.python.value import Value as IRValue
from xls.public.python import runtime_build_actions
from xls.tools.python import eval_helpers

IR_CONVERTER_MAIN_PATH = runfiles.get_path('xls/dslx/ir_converter_main')
EVAL_IR_MAIN_PATH = runfiles.get_path('xls/tools/eval_ir_main')
EVAL_PROC_MAIN_PATH = runfiles.get_path('xls/tools/eval_proc_main')
IR_OPT_MAIN_PATH = runfiles.get_path('xls/tools/opt_main')
CODEGEN_MAIN_PATH = runfiles.get_path('xls/tools/codegen_main')
SIMULATE_MODULE_MAIN_PATH = runfiles.get_path('xls/tools/simulate_module_main')


ArgsBatch = List[List[Value]]
ProcInitValues = List[Value]


def convert_args_batch_to_ir_channel_values(
    args_batch: Optional[ArgsBatch],
    ir_channel_names: Optional[List[str]]) -> Dict[str, Sequence[IRValue]]:
  """Converts args batch to a channel-name-to-ir-values map.

  Args:
    args_batch: The channel values.
    ir_channel_names: List of channel names.

  Returns:
    Dictionary with the key being the channel name and its corresponding value
    being a list of IR values (channel-name-to-ir-values map).

  Raises:
    ValueError: If the number of values in a args batch sample is not equivalent
    to the number of channels.
  """
  all_channel_values: Dict[str, List[Value]] = {k: [] for k in ir_channel_names}
  for channel_values in args_batch:
    if len(channel_values) != len(ir_channel_names):
      raise ValueError(
          'Invalid number of values in args_batch sample; '
          f'expected: {len(ir_channel_names)}, got: {len(channel_values)}')
    for i, value in enumerate(channel_values):
      all_channel_values[ir_channel_names[i]].append(value)

  ir_channel_values: Dict[str, Sequence[IRValue]] = {}
  for key, values in all_channel_values.items():
    ir_channel_values[key] = Value.convert_values_to_ir(values)

  return ir_channel_values


def convert_ir_channel_values_to_channel_values(
    ir_channel_values: Dict[str,
                            Sequence[IRValue]]) -> Dict[str, Sequence[Value]]:
  """Converts a channel-name-to-ir-values map to a channel-name-to-values map.

  Args:
    ir_channel_values: Dictionary with the key being the channel name and its
      corresponding value being a list of IR values (channel-name-to-ir-values
      map).

  Returns:
    Dictionary with the key being the channel name and its corresponding value
    being a list of Values (channel-name-to-values map).
  """
  channel_values: Dict[str,
                       List[Value]] = {k: [] for k in ir_channel_values.keys()}
  for key, values in ir_channel_values.items():
    for value in values:
      channel_values[key].append(interp_value_from_ir_string(value.to_str()))
  return channel_values


class SampleError(XlsError):
  """Error raised if any problem is encountered running the sample.

  These issues can include parsing errors, tools crashing, or result
  miscompares(among others).
  """

  def __init__(self, msg: str, is_timeout: bool = False):
    super().__init__(msg)
    # Whether the sample resulted in a timeout.
    self.is_timeout = is_timeout


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

  def __init__(self, run_dir: str):
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
    ir_channel_names_filename = None
    if smp.args_batch:
      args_filename = self._write_file(
          'args.txt', sample.args_batch_to_text(smp.args_batch))
    if smp.ir_channel_names is not None:
      ir_channel_names_filename = self._write_file(
          'ir_channel_names.txt',
          sample.ir_channel_names_to_text(smp.ir_channel_names))

    self.run_from_files(input_filename, json_options_filename, args_filename,
                        ir_channel_names_filename)

  def run_from_files(self, input_filename: str, json_options_filename: str,
                     args_filename: str, ir_channel_names_filename: str):
    """Runs a sample which is read from files.

    Each filename must be the name of a file (not a full path) which is
    contained in the SampleRunner's run directory.

    Args:
      input_filename: The filename of the sample code.
      json_options_filename: The filename of the JSON-serialized SampleOptions.
      args_filename: The optional filename of the serialized ArgsBatch.
      ir_channel_names_filename: The optional filename of the serialized
        IR channel names.

    Raises:
      SampleError: If an error was encountered.
    """
    logging.vlog(1, 'Running sample in directory %s', self._run_dir)
    logging.vlog(1, 'Reading sample files.')
    options = sample.SampleOptions.from_json(
        self._read_file(json_options_filename))

    self._write_file('revision.txt', revision.get_revision())

    try:
      if options.top_type == sample.TopType.function:
        self._run_function(input_filename, options, args_filename)
      elif options.top_type == sample.TopType.proc:
        self._run_proc(input_filename, options, args_filename,
                       ir_channel_names_filename)
      else:
        raise SampleError(f'Unsupported sample type : {options.top_type}.')

    except Exception as e:  # pylint: disable=broad-except
      # Note: this is a bit of a hack because pybind11 doesn't make it very
      # possible to define custom __str__ on exception types. Our C++ exception
      # types have a field called "message" generally so we look for that.
      msg = (e.message if hasattr(e, 'message') else
             str(e)) + '\n(run dir: {})'.format(self._run_dir)
      logging.exception('Exception when running sample: %s', msg)
      self._write_file('exception.txt', msg)
      raise SampleError(
          msg, is_timeout=isinstance(e, subprocess.TimeoutExpired)) from e

  def _run_function(self, input_filename: str, options: sample.SampleOptions,
                    args_filename: str):
    """Runs a sample with a function as the top which is read from files.

    Each filename must be the name of a file (not a full path) which is
    contained in the SampleRunner's run directory.

    Args:
      input_filename: The filename of the sample code.
      options: The SampleOptions for the sample.
      args_filename: The optional filename of the serialized ArgsBatch.

    Raises:
      SampleError: If an error was encountered.
    """
    input_text = self._read_file(input_filename)
    args_batch: Optional[ArgsBatch] = None
    if args_filename:
      args_batch = sample.parse_args_batch(self._read_file(args_filename))

    # Gather results in an OrderedDict because the first entered result is used
    # as a reference.
    results: Dict[str, Sequence[Value]] = {}
    if options.input_is_dslx:
      if args_batch:
        logging.vlog(1, 'Interpreting DSLX file.')
        with Timer() as t:
          results['interpreted DSLX'] = self._interpret_dslx_function(
              input_text, 'main', args_batch)
        logging.vlog(1, 'Interpreting DSLX complete, elapsed %0.2fs',
                     t.elapsed_ns / 1e9)
        self.timing.interpret_dslx_ns = t.elapsed_ns

      if not options.convert_to_ir:
        return

      with Timer() as t:
        ir_filename = self._dslx_to_ir_function(input_filename, options)
      self.timing.convert_ir_ns = t.elapsed_ns
    else:
      ir_filename = self._write_file('sample.ir', input_text)

    if args_filename is not None:
      # Unconditionally evaluate with the interpreter even if using the
      # JIT. This exercises the interpreter and serves as a reference.
      with Timer() as t:
        results[
            'evaluated unopt IR (interpreter)'] = self._evaluate_ir_function(
                ir_filename, args_filename, False, options)
      self.timing.unoptimized_interpret_ir_ns = t.elapsed_ns

      if options.use_jit:
        with Timer() as t:
          results['evaluated unopt IR (JIT)'] = self._evaluate_ir_function(
              ir_filename, args_filename, True, options)
        self.timing.unoptimized_jit_ns = t.elapsed_ns

    if options.optimize_ir:
      with Timer() as t:
        opt_ir_filename = self._optimize_ir(ir_filename, options)
      self.timing.optimize_ns = t.elapsed_ns

      if args_filename is not None:
        if options.use_jit:
          with Timer() as t:
            results['evaluated opt IR (JIT)'] = self._evaluate_ir_function(
                opt_ir_filename, args_filename, True, options)
          self.timing.optimized_jit_ns = t.elapsed_ns
        with Timer() as t:
          results[
              'evaluated opt IR (interpreter)'] = self._evaluate_ir_function(
                  opt_ir_filename, args_filename, False, options)
        self.timing.optimized_interpret_ir_ns = t.elapsed_ns

      if options.codegen:
        with Timer() as t:
          verilog_filename = self._codegen(opt_ir_filename,
                                           options.codegen_args, options)
        self.timing.codegen_ns = t.elapsed_ns

        if options.simulate:
          assert args_filename is not None
          with Timer() as t:
            results['simulated'] = self._simulate_function(
                verilog_filename, 'module_sig.textproto', args_filename,
                options)
          self.timing.simulate_ns = t.elapsed_ns

    self._compare_results_function(results, args_batch)

  def _run_proc(self, input_filename: str, options: sample.SampleOptions,
                args_filename: str, ir_channel_names_filename: str):
    """Runs a sample with a proc as the top which is read from files.

    Each filename must be the name of a file (not a full path) which is
    contained in the SampleRunner's run directory.

    Args:
      input_filename: The filename of the sample code.
      options: The SampleOptions for the sample.
      args_filename: The optional filename of the serialized ArgsBatch.
      ir_channel_names_filename: The optional filename of the serialized IR
        channel names.

    Raises:
      SampleError: If an error was encountered.
    """
    input_text = self._read_file(input_filename)
    args_batch: Optional[ArgsBatch] = None
    ir_channel_names: Optional[List[str]] = None
    ir_channel_values_filename: Optional[str] = None
    if args_filename:
      args_batch = sample.parse_args_batch(self._read_file(args_filename))
    if ir_channel_names_filename:
      ir_channel_names = sample.parse_ir_channel_names(
          self._read_file(ir_channel_names_filename))
    if args_batch is not None and ir_channel_names is not None:
      ir_channel_values = convert_args_batch_to_ir_channel_values(
          args_batch, ir_channel_names)
      ir_channel_values_file_content = (
          eval_helpers.channel_values_to_string(ir_channel_values))
      ir_channel_values_filename = self._write_file(
          'channel_inputs.txt', ir_channel_values_file_content)

    tick_count = len(args_batch)
    # Special case: When there no inputs for a proc, typically when there are
    # no channels for a proc, tick_count results to 0. Set the tick_count to a
    # non-zero value to execute in the eval proc main (bypasses a restriction on
    # the number of ticks in eval proc main).
    if tick_count == 0:
      tick_count = 1

    # Gather results in an OrderedDict because the first entered result is used
    # as a reference. Note the data is structure with a nested dictionary. The
    # key of the dictionary is the name of the XLS stage being evaluated. The
    # value of the dictionary is another dictionary where the key is the IR
    # channel name. The value of the nested dictionary is a sequence of values
    # corresponding to the channel.
    results: Dict[str, Dict[str, Sequence[Value]]] = {}

    if options.input_is_dslx:
      if args_batch is not None:
        logging.vlog(1, 'Interpreting DSLX file.')
        with Timer() as t:
          results['interpreted DSLX'] = self._interpret_dslx_proc(
              input_text, 'main', args_batch, tick_count)
        logging.vlog(1, 'Interpreting DSLX complete, elapsed %0.2fs',
                     t.elapsed_ns / 1e9)
        self.timing.interpret_dslx_ns = t.elapsed_ns

      if not options.convert_to_ir:
        return

      with Timer() as t:
        ir_filename = self._dslx_to_ir_proc(input_filename, options)
      self.timing.convert_ir_ns = t.elapsed_ns
    else:
      ir_filename = self._write_file('sample.ir', input_text)

    if args_filename is not None:
      # Unconditionally evaluate with the interpreter even if using the
      # JIT. This exercises the interpreter and serves as a reference.
      with Timer() as t:
        results['evaluated unopt IR (interpreter)'] = self._evaluate_ir_proc(
            ir_filename, tick_count, ir_channel_values_filename, False, options)
      self.timing.unoptimized_interpret_ir_ns = t.elapsed_ns

      if options.use_jit:
        with Timer() as t:
          results['evaluated unopt IR (JIT)'] = self._evaluate_ir_proc(
              ir_filename, tick_count, ir_channel_values_filename, True,
              options)
        self.timing.unoptimized_jit_ns = t.elapsed_ns

    if options.optimize_ir:
      with Timer() as t:
        opt_ir_filename = self._optimize_ir(ir_filename, options)
      self.timing.optimize_ns = t.elapsed_ns

      if args_filename is not None:
        if options.use_jit:
          with Timer() as t:
            results['evaluated opt IR (JIT)'] = self._evaluate_ir_proc(
                opt_ir_filename, tick_count, ir_channel_values_filename, True,
                options)
          self.timing.optimized_jit_ns = t.elapsed_ns
        with Timer() as t:
          results['evaluated opt IR (interpreter)'] = self._evaluate_ir_proc(
              opt_ir_filename, tick_count, ir_channel_values_filename, False,
              options)
        self.timing.optimized_interpret_ir_ns = t.elapsed_ns

      if options.codegen:
        with Timer() as t:
          verilog_filename = self._codegen(opt_ir_filename,
                                           options.codegen_args, options)
        self.timing.codegen_ns = t.elapsed_ns

        if options.simulate:
          assert ir_channel_values_filename is not None
          with Timer() as t:
            assert results
            # Pick the first key as a reference to derive the
            # output_channel_counts.
            reference = next(iter(results))
            output_channel_counts = self._get_output_channel_counts(
                results[reference])
            output_channel_counts_str = self._get_output_channel_to_string(
                output_channel_counts)
            results['simulated'] = self._simulate_proc(
                verilog_filename, 'module_sig.textproto',
                ir_channel_values_filename, output_channel_counts_str, options)
          self.timing.simulate_ns = t.elapsed_ns

    self._compare_results_proc(results)

  def _run_command(self, desc: str, args: Sequence[str],
                   options: sample.SampleOptions) -> str:
    """Runs the given commands.

    Args:
      desc: Textual description of what the command is doing. Emitted to stdout.
      args: The command line arguments.
      options: The sample options.

    Returns:
      Stdout of the command.

    Raises:
      subprocess.CalledProcessError: If subprocess returns non-zero code.
      subprocess.TimeoutExpired: If subprocess call times out.
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
          check=False,
          timeout=options.timeout_seconds)

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

  def _write_file(self, filename: str, content: str) -> str:
    """Writes the given content into a named file in the run directory."""
    with open(os.path.join(self._run_dir, filename), 'w') as f:
      f.write(content)
    return filename

  def _read_file(self, filename: str) -> str:
    """Returns the content of the named text file in the run directory."""
    with open(os.path.join(self._run_dir, filename), 'r') as f:
      return f.read()

  def _compare_results_function(self, results: Dict[str, Sequence[Value]],
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

        for i, value in enumerate(values):
          ref_result = results[reference][i]
          if not values_equal(ref_result, value):
            # Bin all of the sources by whether they match the reference or
            # 'values'. This helps identify which of the two is likely
            # correct.
            reference_matches = sorted(
                n for n, v in results.items() if values_equal(v[i], ref_result))
            values_matches = sorted(
                n for n, v in results.items() if values_equal(v[i], value))
            args = '(args unknown)'
            if args_batch:
              args = '; '.join(a.to_ir_str() for a in args_batch[i])
            raise SampleError(f'Result miscompare for sample {i}:'
                              f'\nargs: {args}'
                              f'\n{", ".join(reference_matches)} ='
                              f'\n   {ref_result.to_ir_str()}'
                              f'\n{", ".join(values_matches)} ='
                              f'\n   {value.to_ir_str()}')

  def _compare_results_proc(self, results: Dict[str, Dict[str,
                                                          Sequence[Value]]]):
    """Compares a set of results as for equality.

    Each entry in the map is sequence of Values generated from some source
    (e.g., interpreting the optimized IR). Each sequence of Values is compared
    for equality.

    Args:
      results: Map of result Values.

    Raises:
      SampleError: A miscompare is found.
    """
    if not results:
      return

    # Returns whether the two given values are equal. The IR tools and the
    # verilog simulator produce unsigned values while the DSLX interpreter can
    # produce signed values so compare the results ignoring signedness.
    def values_equal(a: Value, b: Value) -> bool:
      return a.eq(b).is_true()

    stages = sorted(results.keys())
    reference = stages.pop(0)
    all_channel_values_ref = results[reference]
    for name in stages:
      all_channel_values = results[name]
      if len(all_channel_values_ref) != len(all_channel_values):
        raise SampleError(
            f'Results for {reference} has {len(all_channel_values_ref)} '
            f'channel(s), {name} has {len(all_channel_values)} channel(s). '
            f'The IR channel names in {reference} are: '
            f'{list(all_channel_values_ref.keys())}.'
            f'The IR channel names in {name} are: '
            f'{list(all_channel_values.keys())}.')
      for channel_name_ref, channel_values_ref in all_channel_values_ref.items(
      ):
        channel_values = all_channel_values.get(channel_name_ref)
        if channel_values is None:
          raise SampleError(
              f'A channel named {channel_name_ref} is present in {reference}, '
              f'but it is not present in {name}.')
        if len(channel_values_ref) != len(channel_values):
          raise SampleError(
              f'In {reference}, channel \'{channel_name_ref}\' has '
              f'{len(channel_values_ref)} entries. However, in {name}, '
              f'channel \'{channel_name_ref}\' has {len(channel_values)} '
              'entries.')
        for index, (value, value_ref) in enumerate(
            zip(channel_values, channel_values_ref)):
          if not values_equal(value_ref, value):
            raise SampleError(
                f'In {reference}, at position {index} channel '
                f'\'{channel_name_ref}\' has value {value_ref}. However, in '
                f'{name}, the value is {value}.')

  def _interpret_dslx_function(self, text: str, top_name: str,
                               args_batch: ArgsBatch) -> Tuple[Value, ...]:
    """Interprets a DSLX module with a function as the top returns the result Values.
    """
    import_data = create_import_data.create_import_data(
        runtime_build_actions.get_default_dslx_stdlib_path(), []
    )
    tm = parse_and_typecheck.parse_and_typecheck(
        text, 'sample.x', 'sample', import_data
    )
    converted_args_batch = interpreter.convert_function_kwargs(
        top_name, import_data, tm, args_batch
    )
    dslx_results = interpreter.run_function_batched(
        top_name, import_data, tm, converted_args_batch
    )
    self._write_file('sample.x.results',
                     '\n'.join(r.to_ir_str() for r in dslx_results))
    return tuple(dslx_results)

  def _interpret_dslx_proc(self, text: str, top_name: str,
                           args_batch: ArgsBatch,
                           tick_count: int) -> Dict[str, Sequence[Value]]:
    """Interprets a DSLX module with proc as the top returns the result Values.
    """
    import_data = create_import_data.create_import_data(
        runtime_build_actions.get_default_dslx_stdlib_path(), []
    )
    tm = parse_and_typecheck.parse_and_typecheck(
        text, 'sample.x', 'sample', import_data
    )
    converted_channel_values = interpreter.convert_channel_values(
        top_name, import_data, tm, args_batch
    )
    dslx_results = interpreter.run_proc(
        top_name, import_data, tm, converted_channel_values, tick_count
    )

    ir_channel_values: Dict[str, Sequence[IRValue]] = {}
    for key, values in dslx_results.items():
      ir_channel_values[key] = Value.convert_values_to_ir(values)
    self._write_file('sample.x.results',
                     eval_helpers.channel_values_to_string(ir_channel_values))
    return dslx_results

  def _parse_values(self, s: str) -> Tuple[Value, ...]:
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
        interp_value_from_ir_string(line.strip())
        for line in s.split('\n')
        if line.strip())

  def _evaluate_ir_function(self, ir_filename: str, args_filename: str,
                            use_jit: bool,
                            options: sample.SampleOptions) -> Tuple[Value, ...]:
    """Evaluate the IR file with a function as the top and returns the result Values.
    """
    results_text = self._run_command(
        'Evaluating IR file ({}): {}'.format(
            'JIT' if use_jit else 'interpreter', ir_filename),
        (EVAL_IR_MAIN_PATH, '--input_file=' + args_filename,
         '--use_llvm_jit' if use_jit else '--nouse_llvm_jit', ir_filename),
        options)
    self._write_file(ir_filename + '.results', results_text)
    return self._parse_values(results_text)

  def _evaluate_ir_proc(
      self, ir_filename: str, tick_count: int,
      ir_channel_values_filename: Optional[str], use_jit: bool,
      options: sample.SampleOptions) -> Dict[str, Sequence[Value]]:
    """Evaluate the IR file with a proc as the top and returns the result Values.
    """
    evaluation_type = 'JIT' if use_jit else 'interpreter'
    desc = f'Evaluating IR file ({evaluation_type}): {ir_filename}'
    backend_type = 'serial_jit' if use_jit else 'ir_interpreter'
    args = (EVAL_PROC_MAIN_PATH,
            '--inputs_for_all_channels=' + ir_channel_values_filename,
            '--ticks=' + str(tick_count), '--backend=' + backend_type,
            ir_filename)
    results_text = self._run_command(desc, args, options)
    self._write_file(ir_filename + '.results', results_text)

    ir_channel_values = eval_helpers.parse_channel_values(
        results_text, tick_count)
    return convert_ir_channel_values_to_channel_values(ir_channel_values)

  def _dslx_to_ir_function(self, dslx_filename: str,
                           options: sample.SampleOptions) -> str:
    """Converts the DSLX file to an IR file with a function as the top whose filename is returned.
    """
    args = [IR_CONVERTER_MAIN_PATH]
    if options.ir_converter_args:
      args.extend(options.ir_converter_args)
    args.append(dslx_filename)
    ir_text = self._run_command('Converting DSLX to IR', args, options)
    logging.vlog(3, 'Unoptimized IR:\n%s', ir_text)
    return self._write_file('sample.ir', ir_text)

  def _dslx_to_ir_proc(self, dslx_filename: str,
                       options: sample.SampleOptions) -> str:
    """Converts the DSLX file to an IR file with a proc as the top whose filename is returned.
    """
    args = [IR_CONVERTER_MAIN_PATH]
    if options.ir_converter_args:
      args.extend(options.ir_converter_args)
    args.append(dslx_filename)
    ir_text = self._run_command('Converting DSLX to IR', args, options)
    logging.vlog(3, 'Unoptimized IR:\n%s', ir_text)
    return self._write_file('sample.ir', ir_text)

  def _optimize_ir(self, ir_filename: str,
                   options: sample.SampleOptions) -> str:
    """Optimizes the IR file and returns the resulting filename."""
    opt_ir_text = self._run_command('Optimizing IR',
                                    (IR_OPT_MAIN_PATH, ir_filename), options)
    logging.vlog(3, 'Optimized IR:\n%s', opt_ir_text)
    return self._write_file('sample.opt.ir', opt_ir_text)

  def _codegen(self, ir_filename: str, codegen_args: Sequence[str],
               options: sample.SampleOptions) -> str:
    """Generates Verilog from the IR file and return the Verilog filename."""
    args = [
        CODEGEN_MAIN_PATH, '--output_signature_path=module_sig.textproto',
        '--delay_model=unit'
    ]
    args.extend(codegen_args)
    args.append(ir_filename)
    verilog_text = self._run_command('Generating Verilog', args, options)
    logging.vlog(3, 'Verilog:\n%s', verilog_text)
    return self._write_file(
        'sample.sv' if options.use_system_verilog else 'sample.v', verilog_text)

  def _simulate_function(self, verilog_filename: str, module_sig_filename: str,
                         args_filename: str,
                         options: sample.SampleOptions) -> Tuple[Value, ...]:
    """Simulates the Verilog file representing a function and returns the results Values.
    """
    simulator_args = [
        SIMULATE_MODULE_MAIN_PATH,
        '--signature_file=' + module_sig_filename,
        '--args_file=' + args_filename
    ]
    if options.simulator:
      simulator_args.append('--verilog_simulator=' + options.simulator)
    simulator_args.append(verilog_filename)

    check_simulator.check_simulator(options.simulator)

    results_text = self._run_command(f'Simulating Verilog {verilog_filename}',
                                     simulator_args, options)
    self._write_file(verilog_filename + '.results', results_text)
    return self._parse_values(results_text)

  def _get_output_channel_counts(
      self, output_channel_values: Dict[str,
                                        Sequence[Value]]) -> Dict[str, int]:
    """Returns a output-channel-count map from an output-channel-values map."""
    output_channel_counts: Dict[str, int] = {}
    for channel_name, channel_values in output_channel_values.items():
      output_channel_counts[channel_name] = len(channel_values)
    return output_channel_counts

  def _get_output_channel_to_string(
      self, output_channel_values: Dict[str, int]) -> str:
    """Returns a string representation of the output-channel-count map.

    The string format is output_channel_name=count for each entry in the
    map. The entries of the map are comma separated. For example, given an
    output-channel-count map:

    {{foo, 42}{bar,64}}

    the string representation is:

    foo=42,bar=64.

    Args:
      output_channel_values: Dictionary with the key being the channel name and
        its corresponding value a count of values (output-channel-count map).

    Returns:
      A string representation of the output-channel-count map.
    """
    output_channel_counts_strings = []
    for channel_name, count in output_channel_values.items():
      output_channel_counts_strings.append(f'{channel_name}={count}')
    return ','.join(output_channel_counts_strings)

  def _simulate_proc(
      self, verilog_filename: str, module_sig_filename: str,
      ir_channel_values_filename: str, output_channel_counts: str,
      options: sample.SampleOptions) -> Dict[str, Sequence[Value]]:
    """Simulates the Verilog file representing a proc and returns the results Values.
    """
    simulator_args = [
        SIMULATE_MODULE_MAIN_PATH, '--signature_file=' + module_sig_filename,
        '--channel_values_file=' + ir_channel_values_filename,
        '--output_channel_counts=' + output_channel_counts
    ]
    if options.simulator:
      simulator_args.append('--verilog_simulator=' + options.simulator)
    simulator_args.append(verilog_filename)

    check_simulator.check_simulator(options.simulator)

    results_text = self._run_command(f'Simulating Verilog {verilog_filename}',
                                     simulator_args, options)
    self._write_file(verilog_filename + '.results', results_text)

    ir_channel_values = eval_helpers.parse_channel_values(results_text)
    return convert_ir_channel_values_to_channel_values(ir_channel_values)

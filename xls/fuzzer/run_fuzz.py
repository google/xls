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

"""Fuzzer generate-and-compare loop."""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Tuple, Text, Optional

from absl import logging
import termcolor

from xls.common import runfiles
from xls.fuzzer import sample_runner
from xls.fuzzer import sample_summary_pb2
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python import cpp_sample as sample

SAMPLE_RUNNER_MAIN_PATH = runfiles.get_path('xls/fuzzer/sample_runner_main')
IR_MINIMIZER_MAIN_PATH = runfiles.get_path('xls/tools/ir_minimizer_main')
SUMMARIZE_IR_MAIN_PATH = runfiles.get_path('xls/fuzzer/summarize_ir_main')
FIND_FAILING_INPUT_MAIN = runfiles.get_path(
    'xls/fuzzer/find_failing_input_main')


def _write_to_file(dir_path: Text,
                   filename: Text,
                   content: Text,
                   executable: bool = False):
  """Writes the content into a file of the given name in the directory."""
  path = os.path.join(dir_path, filename)
  with open(path, 'w') as f:
    f.write(content)
  if executable:
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR)


def _write_ir_summaries(run_dir: str,
                        timing: sample_summary_pb2.SampleTimingProto,
                        summary_path: str):
  """Appends IR summaries of IR files in the run dir to the summary file."""
  args = []

  unoptimized_path = os.path.join(run_dir, 'sample.ir')
  if os.path.exists(unoptimized_path):
    args.append('--unoptimized_ir=' + unoptimized_path)

  optimized_path = os.path.join(run_dir, 'sample.opt.ir')
  if os.path.exists(optimized_path):
    args.append('--optimized_ir=' + optimized_path)
  if not args:
    return

  subprocess.run(
      [
          SUMMARIZE_IR_MAIN_PATH,
          '--logtostderr',
          '--minloglevel=2',
          '--summary_file=' + summary_path,
          '--timing=' + str(timing),
      ] + args,
      check=False)


def run_sample(smp: sample.Sample,
               run_dir: Text,
               summary_file: Optional[Text] = None,
               generate_sample_ns: Optional[int] = None):
  """Runs the given sample in the given directory.

  Args:
    smp: Sample to run.
    run_dir: Directory to run the sample in. The directory should exist and be
      empty.
    summary_file: The (optional) file to append sample summary.
    generate_sample_ns: The (optional) time in nanoseconds to generate the
      sample. Recorded in the summary file, if given.

  Raises:
    sample_runner.SampleError: on any non-zero status from the sample runner.

  """
  start = time.time()

  _write_to_file(run_dir, 'sample.x', smp.input_text)
  _write_to_file(run_dir, 'options.json', smp.options.to_json())
  if smp.args_batch:
    _write_to_file(run_dir, 'args.txt',
                   sample.args_batch_to_text(smp.args_batch))

  # Create a script named 'run.sh' for rerunning the sample.
  args = [
      SAMPLE_RUNNER_MAIN_PATH, '--logtostderr', '--input_file=sample.x',
      '--options_file=options.json'
  ]
  if smp.args_batch:
    args.append('--args_file=args.txt')
  args.append(run_dir)
  _write_to_file(
      run_dir,
      'run.sh',
      f'#!/bin/sh\n\n{subprocess.list2cmdline(args)}\n',
      executable=True)
  logging.vlog(1, 'Starting to run sample')
  logging.vlog(2, smp.input_text)
  runner = sample_runner.SampleRunner(run_dir)
  runner.run_from_files('sample.x', 'options.json', 'args.txt')
  timing = runner.timing

  timing.total_ns = int((time.time() - start) * 1e9)
  if generate_sample_ns:
    # The sample generation time, if given, is not part of the measured total
    # time, so add it in.
    timing.total_ns += generate_sample_ns
    timing.generate_sample_ns = generate_sample_ns

  logging.vlog(1, 'Completed running sample, elapsed: %0.2fs',
               time.time() - start)

  if summary_file:
    _write_ir_summaries(run_dir, timing, summary_file)


def minimize_ir(smp: sample.Sample,
                run_dir: Text,
                inject_jit_result: Optional[Text] = None) -> Optional[Text]:
  """Tries to minimize the IR of the given sample in the run directory.

  Writes a test script into the run_directory for testing the IR for the
  failure. Passes this test script to ir_minimizer_main to try to create a
  minimal IR sample.

  Args:
    smp: The sample to try to minimize.
    run_dir: The run directory the sample was run in.
    inject_jit_result: For testing only. Value to produce as the JIT result.
  Returns:
    The path to the minimized IR file (created in the run directory), or None if
    minimization was not possible
  """
  if os.path.exists(os.path.join(run_dir, 'sample.ir')):
    # First try to minimize using the sample runner binary as the minimization
    # test.
    ir_minimize_options = smp.options.replace(input_is_dslx=False)
    _write_to_file(run_dir, 'ir_minimizer.options.json',
                   ir_minimize_options.to_json())
    # Generate the sample runner script. The script should return 0 (success) if
    # the sample fails so invert the return code of the invocation of
    # sample_runner_main with '!'.
    args = [
        SAMPLE_RUNNER_MAIN_PATH, '--logtostderr',
        '--options_file=ir_minimizer.options.json', '--args_file=args.txt',
        '--input_file=$1'
    ]
    _write_to_file(
        run_dir,
        'ir_minimizer_test.sh',
        f'#!/bin/sh\n! {" ".join(args)}',
        executable=True)
    comp = subprocess.run([
        IR_MINIMIZER_MAIN_PATH, '--logtostderr',
        '--test_executable=ir_minimizer_test.sh', 'sample.ir'
    ],
                          cwd=run_dir,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          check=False)
    if comp.returncode == 0:
      minimized_ir_path = os.path.join(run_dir, 'minimized.ir')
      with open(minimized_ir_path, 'wb') as f:
        f.write(comp.stdout)
      return minimized_ir_path

    if smp.options.use_jit:
      # Next try to minimize assuming the underlying cause was a JIT mismatch.
      # The IR minimizer binary has special machinery for reducing these kinds
      # of failures. The minimization occurs in two steps:
      # (1) Find an input that results in a JIT vs interpreter mismatch (if any)
      # (2) Run the minimization tool using this input as the test.
      extra_args = ['--test_only_inject_jit_result=' +
                    inject_jit_result] if inject_jit_result else []
      comp = subprocess.run(
          [FIND_FAILING_INPUT_MAIN, '--input_file=args.txt', 'sample.ir'] +
          extra_args,
          cwd=run_dir,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          check=False)
      if comp.returncode == 0:
        # A failing input for JIT vs interpreter was found
        failed_input = comp.stdout.decode('utf-8')
        comp = subprocess.run(
            [
                IR_MINIMIZER_MAIN_PATH, '--logtostderr', '--test_llvm_jit',
                '--use_optimization_pipeline', '--input=' + failed_input,
                'sample.ir'
            ] + extra_args,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False)
        if comp.returncode == 0:
          minimized_ir_path = os.path.join(run_dir, 'minimized.ir')
          with open(minimized_ir_path, 'wb') as f:
            f.write(comp.stdout)
          return minimized_ir_path

  return None


def run_fuzz(
    rng: ast_generator.RngState,
    ast_generator_options: ast_generator.AstGeneratorOptions,
    calls_per_sample: int,
    save_temps: bool,
    sample_count: int,
    codegen: bool,
    simulate: bool = False,
    return_samples: bool = False) -> Optional[Tuple[sample.Sample, ...]]:
  """Runs a fuzzing loop for "sample_count" samples."""
  samples = []
  for i in range(sample_count):
    smp = ast_generator.generate_sample(
        ast_generator_options, calls_per_sample,
        sample.SampleOptions(
            input_is_dslx=True,
            ir_converter_args=['--top=main'],
            convert_to_ir=True,
            optimize_ir=True,
            codegen=codegen,
            simulate=simulate), rng)

    if return_samples:
      samples.append(smp)

    termcolor.cprint('=== Sample {}'.format(i), color='yellow', file=sys.stderr)
    for lineno, line in enumerate(smp.input_text.splitlines(), 1):
      print('{:03d}: {}'.format(lineno, line))
    print(smp.to_crasher('<sample {}>'.format(i)), file=sys.stderr)

    sample_dir = tempfile.mkdtemp('run_fuzz_')
    try:
      run_sample(smp, sample_dir)
    except Exception as e:
      print(smp.to_crasher(str(e)), file=sys.stderr)
      raise

    if not save_temps:
      shutil.rmtree(sample_dir)

  if return_samples:
    return tuple(samples)

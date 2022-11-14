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

import datetime
import hashlib
import os
import stat
import subprocess
import time
from typing import Text, Optional

from absl import logging

from xls.common import gfile
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
  # Create a script named 'run.sh' for rerunning the sample.
  args = [
      SAMPLE_RUNNER_MAIN_PATH, '--logtostderr', '--input_file=sample.x',
      '--options_file=options.json'
  ]

  _write_to_file(run_dir, 'sample.x', smp.input_text)
  _write_to_file(run_dir, 'options.json', smp.options.to_json())
  args_filename = None
  if smp.args_batch:
    args_filename = 'args.txt'
    _write_to_file(run_dir, args_filename,
                   sample.args_batch_to_text(smp.args_batch))
    args.append('--args_file=args.txt')
  ir_channel_names_filename = None
  if smp.ir_channel_names is not None:
    ir_channel_names_filename = 'ir_channel_names.txt'
    _write_to_file(run_dir, ir_channel_names_filename,
                   sample.ir_channel_names_to_text(smp.ir_channel_names))
    args.append('--ir_channel_names_file=ir_channel_names.txt')
  args.append(run_dir)
  _write_to_file(
      run_dir,
      'run.sh',
      f'#!/bin/sh\n\n{subprocess.list2cmdline(args)}\n',
      executable=True)
  logging.vlog(1, 'Starting to run sample')
  logging.vlog(2, smp.input_text)
  runner = sample_runner.SampleRunner(run_dir)
  runner.run_from_files('sample.x', 'options.json', args_filename,
                        ir_channel_names_filename)
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
                inject_jit_result: Optional[Text] = None,
                timeout: Optional[datetime.timedelta] = None) -> Optional[Text]:
  """Tries to minimize the IR of the given sample in the run directory.

  Writes a test script into the run_directory for testing the IR for the
  failure. Passes this test script to ir_minimizer_main to try to create a
  minimal IR sample.

  Args:
    smp: The sample to try to minimize.
    run_dir: The run directory the sample was run in.
    inject_jit_result: For testing only. Value to produce as the JIT result.
    timeout: Timeout for running the minimizer.

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
    basename = os.path.basename(IR_MINIMIZER_MAIN_PATH)
    stderr_path = os.path.join(run_dir, basename + '.stderr')
    try:
      with open(stderr_path, 'w') as f_stderr:
        comp = subprocess.run([
            IR_MINIMIZER_MAIN_PATH, '--logtostderr',
            '--test_executable=ir_minimizer_test.sh', 'sample.ir'
        ],
                              cwd=run_dir,
                              stdout=subprocess.PIPE,
                              stderr=f_stderr,
                              check=False,
                              timeout=timeout.seconds if timeout else None)
    except subprocess.TimeoutExpired:
      return None
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
      basename = os.path.basename(FIND_FAILING_INPUT_MAIN)
      stderr_path = os.path.join(run_dir, basename + '.stderr')
      try:
        with open(stderr_path, 'w') as f_stderr:
          comp = subprocess.run(
              [FIND_FAILING_INPUT_MAIN, '--input_file=args.txt', 'sample.ir'] +
              extra_args,
              cwd=run_dir,
              stdout=subprocess.PIPE,
              stderr=f_stderr,
              check=False,
              timeout=timeout.seconds if timeout else None)
      except subprocess.TimeoutExpired:
        return None
      if comp.returncode == 0:
        # A failing input for JIT vs interpreter was found
        failed_input = comp.stdout.decode('utf-8')
        basename = os.path.basename(IR_MINIMIZER_MAIN_PATH)
        stderr_path = os.path.join(run_dir, basename + '_jit.stderr')
        try:
          with open(stderr_path, 'w') as f_stderr:
            comp = subprocess.run(
                [
                    IR_MINIMIZER_MAIN_PATH, '--logtostderr', '--test_llvm_jit',
                    '--use_optimization_pipeline', '--input=' + failed_input,
                    'sample.ir'
                ] + extra_args,
                cwd=run_dir,
                stdout=subprocess.PIPE,
                stderr=f_stderr,
                check=False,
                timeout=timeout.seconds if timeout else None)
        except subprocess.TimeoutExpired:
          return None
        if comp.returncode == 0:
          minimized_ir_path = os.path.join(run_dir, 'minimized.ir')
          with open(minimized_ir_path, 'wb') as f:
            f.write(comp.stdout)
          return minimized_ir_path

  return None


def _save_crasher(run_dir: str, smp: sample.Sample,
                  exception: sample_runner.SampleError,
                  crasher_dir: str) -> str:
  """Saves the sample into a new directory in the crasher directory."""
  digest = hashlib.sha256(smp.input_text.encode('utf-8')).hexdigest()[:8]
  sample_crasher_dir = os.path.join(crasher_dir, digest)
  logging.info('Saving crasher to %s', sample_crasher_dir)
  gfile.recursively_copy_dir(
      run_dir, sample_crasher_dir, preserve_file_mask=True)
  with gfile.open(os.path.join(sample_crasher_dir, 'exception.txt'), 'w') as f:
    f.write(str(exception))
  crasher_path = os.path.join(
      sample_crasher_dir,
      'crasher_{}_{}.x'.format(datetime.date.today().strftime('%Y-%m-%d'),
                               digest[:4]))
  with gfile.open(crasher_path, 'w') as f:
    f.write(smp.to_crasher(str(exception)))
  return sample_crasher_dir


def generate_sample_and_run(
    rng: ast_generator.ValueGenerator,
    ast_generator_options: ast_generator.AstGeneratorOptions,
    sample_options: sample.SampleOptions,
    run_dir: str,
    crasher_dir: Optional[str] = None,
    summary_file: Optional[str] = None,
    force_failure: bool = False) -> sample.Sample:
  """Generates and runs a fuzzing sample."""
  with sample_runner.Timer() as t:
    smp = ast_generator.generate_sample(ast_generator_options, sample_options,
                                        rng)
  try:
    run_sample(
        smp,
        run_dir,
        summary_file=summary_file,
        generate_sample_ns=t.elapsed_ns)
    if force_failure:
      raise sample_runner.SampleError('Forced sample failure.')
  except sample_runner.SampleError as e:
    logging.error('Sample failed: %s', str(e))
    if crasher_dir is not None:
      sample_crasher_dir = _save_crasher(run_dir, smp, e, crasher_dir)
      if not e.is_timeout:
        logging.info('Attempting to minimize IR...')
        ir_minimized = minimize_ir(
            smp,
            sample_crasher_dir,
            timeout=datetime.timedelta(seconds=sample_options.timeout_seconds)
            if sample_options.timeout_seconds else None)
        if ir_minimized:
          logging.info('...minimization successful.')
        else:
          logging.info('...minimization failed.')
      raise e

  return smp

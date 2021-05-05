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
"""Tests for xls.fuzzer.run_fuzz_multiprocess."""

import os
import subprocess

from xls.common import runfiles
from xls.common import test_base

RUN_FUZZ_MULTIPROCESS_PATH = runfiles.get_path(
    'xls/fuzzer/run_fuzz_multiprocess')


class RunFuzzMultiprocessTest(test_base.TestCase):

  def test_two_samples(self):
    crasher_path = self.create_tempdir().full_path
    samples_path = self.create_tempdir().full_path

    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH, '--seed=42', '--crash_path=' + crasher_path,
        '--save_temps_path=' + samples_path, '--sample_count=2',
        '--calls_per_sample=3', '--worker_count=1'
    ])
    # Crasher path should contain a single file 'test'.
    self.assertSequenceEqual(os.listdir(crasher_path), ('test',))

    # Sample path should have two samples in it.
    self.assertSequenceEqual(sorted(os.listdir(samples_path)), ('0', '1'))

    # Validate sample 1 directory.
    sample1_contents = os.listdir(os.path.join(samples_path, '1'))
    self.assertIn('sample.x', sample1_contents)
    self.assertIn('sample.x.results', sample1_contents)
    self.assertIn('args.txt', sample1_contents)
    self.assertIn('sample.ir', sample1_contents)
    self.assertIn('sample.ir.results', sample1_contents)
    self.assertIn('sample.opt.ir', sample1_contents)
    self.assertIn('sample.opt.ir.results', sample1_contents)

    # Codegen was not enabled so there should be no Verilog file.
    self.assertNotIn('sample.v', sample1_contents)

    # Args file should have three lines in it.
    with open(os.path.join(samples_path, '1', 'args.txt')) as f:
      self.assertEqual(len(f.read().strip().splitlines()), 3)

  def test_multiple_workers(self):
    crasher_path = self.create_tempdir().full_path
    samples_path = self.create_tempdir().full_path

    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH, '--seed=42', '--crash_path=' + crasher_path,
        '--save_temps_path=' + samples_path, '--sample_count=20',
        '--calls_per_sample=3', '--worker_count=10'
    ])

    # Sample path should have 20 samples in it.
    self.assertSequenceEqual(
        sorted(os.listdir(samples_path)), sorted((str(i) for i in range(20))))

  def test_codegen_and_simulate(self):
    crasher_path = self.create_tempdir().full_path
    samples_path = self.create_tempdir().full_path

    subprocess.check_call([
        RUN_FUZZ_MULTIPROCESS_PATH, '--seed=42', '--crash_path=' + crasher_path,
        '--save_temps_path=' + samples_path, '--sample_count=2',
        '--calls_per_sample=3', '--worker_count=1', '--codegen', '--simulate',
        '--nouse_system_verilog'
    ])
    # Validate sample 1 directory.
    sample1_contents = os.listdir(os.path.join(samples_path, '1'))
    self.assertIn('sample.x', sample1_contents)
    self.assertIn('sample.x.results', sample1_contents)
    self.assertIn('args.txt', sample1_contents)
    self.assertIn('sample.ir', sample1_contents)
    self.assertIn('sample.ir.results', sample1_contents)
    self.assertIn('sample.opt.ir', sample1_contents)
    self.assertIn('sample.opt.ir.results', sample1_contents)
    # Directory should have verilog and simulation results.
    self.assertIn('sample.v', sample1_contents)
    self.assertIn('sample.v.results', sample1_contents)


if __name__ == '__main__':
  test_base.main()

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

"""Tests for xls.tools.codegen_main."""

import subprocess

from xls.common import runfiles
from xls.common import test_base

OPT_MAIN_PATH = runfiles.get_path('xls/tools/opt_main')

ADD_ZERO_IR = """package add_zero

top fn add_zero(x: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0, id=1)
  ret add.2: bits[32] = add(x, literal.1, id=2)
}
"""

DEAD_FUNCTION_IR = """package add_zero

fn dead_function(x: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret add.2: bits[32] = add(x, literal.1)
}

top fn main(x: bits[32]) -> bits[32] {
  literal.3: bits[32] = literal(value=0)
  ret add.4: bits[32] = add(x, literal.3)
}
"""

ADD_LITERAL_IR = """package add_literal

top fn add_zero(x: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0xfff_0000)
  ret add.2: bits[32] = add(x, literal.1)
}
"""


class OptMainTest(test_base.TestCase):

  def test_no_options(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path]
    ).decode('utf-8')

    # The add with zero should be eliminated.
    self.assertIn('ret x', optimized_ir)

  def test_with_vlog(self):
    # Checks that enabling vlog doesn't crash.
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '-v=5', ir_file.full_path]
    ).decode('utf-8')

    # The add with zero should be eliminated.
    self.assertIn('ret x', optimized_ir)

  def test_run_only_basic_simp_and_dce_passes(self):
    # NB This used to test a deprecated --run_only_passes flag that is
    # superseeded by the --passes flag.
    ir_file = self.create_tempfile(content=DEAD_FUNCTION_IR)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--passes=basic_simp dce', ir_file.full_path]
    ).decode('utf-8')

    # The add with zero should be eliminated.
    self.assertIn('add(', ADD_ZERO_IR)
    self.assertNotIn('add(', optimized_ir)

    # Without running DFE, the dead function should remain in the IR.
    self.assertIn('dead_function', optimized_ir)

  def test_skip_dfe(self):
    ir_file = self.create_tempfile(content=DEAD_FUNCTION_IR)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path]
    ).decode('utf-8')

    # Without skipping DFE, the dead function should be removed.
    self.assertNotIn('dead_function', optimized_ir)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--skip_passes=dfe', ir_file.full_path]
    ).decode('utf-8')

    # Skipping DFE should leave the dead function in the IR.
    self.assertIn('dead_function', optimized_ir)

  def test_opt_level(self):
    ir_file = self.create_tempfile(content=ADD_LITERAL_IR)

    # Without specifying opt-level, the optimizer should run at maximum level
    # and convert the add with literal 0xffff_0000 into something like:
    #
    #   concat(add(0xffff, x[16:32]), x[0:16])
    #
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path]
    ).decode('utf-8')
    self.assertIn('bits[16] = add', optimized_ir)
    self.assertIn('concat', optimized_ir)

    # Opt_level 3 should produce the same results as without opt_level
    # specified.
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--opt_level=3', ir_file.full_path]
    ).decode('utf-8')
    self.assertIn('bits[16] = add', optimized_ir)
    self.assertIn('concat', optimized_ir)

    # At opt_level 1 the full width add should remain.
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--opt_level=1', ir_file.full_path]
    ).decode('utf-8')
    self.assertIn('bits[32] = add', optimized_ir)
    self.assertNotIn('concat', optimized_ir)

  def test_explicit_pipeline(self):
    """Check that basic_simp gets the add-zero gone."""
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path, '--passes', 'basic_simp dce']
    ).decode('utf-8')
    self.assertNotIn('bits[32] = add', optimized_ir)

  def test_explicit_single_pass_pipeline(self):
    """Check that dce is not run."""
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path, '--passes', 'basic_simp']
    ).decode('utf-8')
    # add is not removed since the DCE was not run.
    self.assertIn('bits[32] = add', optimized_ir)

  # TODO: https://github.com/google/xls/issues/1245 - Get an example that can
  # see the fixed-point pipeline doing something.

  def test_bisect_limit(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        ir_file.full_path,
        '--passes',
        'dce dce dce dce basic_simp',
        '--passes_bisect_limit',
        '3',
    ]).decode('utf-8')
    # No change since basic_simp is not run
    self.assertEqual(optimized_ir, ADD_ZERO_IR)

  def test_bisect_limit_allows_changes(self):
    """Check that dce is not run after basic simp."""
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        ir_file.full_path,
        '--passes',
        'dce basic_simp dce',
        '--passes_bisect_limit',
        '2',
    ]).decode('utf-8')
    # add is not removed since the DCE was not run.
    self.assertIn('bits[32] = add', optimized_ir)


if __name__ == '__main__':
  test_base.main()

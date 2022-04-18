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

"""Tests for xls.codegen.python.pipeline_generator."""

import sys

from xls.codegen.python import pipeline_generator
from xls.common.python import init_xls
from absl.testing import absltest
from xls.ir.python import function_builder
from xls.ir.python import package


def setUpModule():
  # This is required so that module initializers are called including those
  # which register delay models.
  init_xls.init_xls(sys.argv)


class PipelineGeneratorTest(absltest.TestCase):

  def test_pipeline_generator_with_n_stages(self):
    pkg = package.Package('pname')
    fb = function_builder.FunctionBuilder('main', pkg)
    fb.add_param('x', pkg.get_bits_type(32))
    fb.build()
    pkg.set_top_by_name('main')

    module_signature = pipeline_generator.generate_pipelined_module_with_n_stages(
        pkg, 5, 'foo')

    self.assertIn('module foo', module_signature.verilog_text)
    self.assertIn('p0_x', module_signature.verilog_text)
    self.assertIn('p1_x', module_signature.verilog_text)
    self.assertIn('p2_x', module_signature.verilog_text)
    self.assertIn('p3_x', module_signature.verilog_text)
    self.assertIn('p4_x', module_signature.verilog_text)

  def test_pipeline_generator_with_clock_period(self):
    pkg = package.Package('pname')
    fb = function_builder.FunctionBuilder('main', pkg)
    fb.add_param('x', pkg.get_bits_type(32))
    fb.build()
    pkg.set_top_by_name('main')

    module_signature = pipeline_generator.generate_pipelined_module_with_clock_period(
        pkg, 100, 'bar')

    self.assertIn('module bar', module_signature.verilog_text)
    self.assertIn('p0_x', module_signature.verilog_text)
    self.assertIn('p1_x', module_signature.verilog_text)


if __name__ == '__main__':
  absltest.main()

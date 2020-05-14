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

"""Tests for xls.codegen.python.pipeline_generator."""

import sys

from xls.codegen.python import pipeline_generator
from xls.common.python import init_xls
from absl.testing import absltest
from xls.ir.python import bits as bits_mod
from xls.ir.python import function_builder
from xls.ir.python import package


def setUpModule():
  # This is required so that module initializers are called including those
  # which register delay models.
  init_xls.init_xls(sys.argv)


class PipelineGeneratorTest(absltest.TestCase):

  def test_pipeline_generator(self):
    pkg = package.Package('pname')
    fb = function_builder.FunctionBuilder('main', pkg)
    fb.add_literal_bits(bits_mod.UBits(value=2, bit_count=32))
    fb.build()

    module_signature = pipeline_generator.schedule_and_generate_pipelined_module(
        pkg, 10)

    self.assertIn('module main', module_signature.verilog_text)


if __name__ == '__main__':
  absltest.main()

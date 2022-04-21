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
#

"""Tests for xls.passes.python.standard_pipeline."""

import sys

from xls.common.python import init_xls
from xls.ir.python import bits as bits_mod
from xls.ir.python import function_builder
from xls.ir.python import package
from xls.passes.python import standard_pipeline
from absl.testing import absltest


def setUpModule():
  # This is required so that module initializers are called including those
  # which register delay models.
  init_xls.init_xls(sys.argv)


class StandardPipelineTest(absltest.TestCase):

  def test_standard_pipeline(self):
    pkg = package.Package('pname')
    fb = function_builder.FunctionBuilder('main', pkg)
    fb.add_literal_bits(bits_mod.UBits(value=2, bit_count=32))
    fb.build()

    self.assertFalse(standard_pipeline.run_standard_pass_pipeline(pkg))


if __name__ == '__main__':
  absltest.main()

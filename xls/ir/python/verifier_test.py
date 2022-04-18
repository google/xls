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

"""Tests for xls.ir.python.verifier."""

from xls.ir.python import bits
from xls.ir.python import function_builder
from xls.ir.python import package
from xls.ir.python import value as ir_value
from xls.ir.python import verifier
from absl.testing import absltest


class VerifierTest(absltest.TestCase):

  def test_verify_package(self):
    pkg = package.Package('pkg')
    verifier.verify_package(pkg)

  def test_verify_function(self):
    pkg = package.Package('pname')
    builder = function_builder.FunctionBuilder('f_name', pkg)
    builder.add_param('x', pkg.get_bits_type(32))
    builder.add_literal_value(ir_value.Value(bits.UBits(7, 8)))
    fn = builder.build()

    verifier.verify_function(fn)


if __name__ == '__main__':
  absltest.main()

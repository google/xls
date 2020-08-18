# Lint as: python3
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
"""Tests for xls.solvers.lec_characterizer."""

import portpicker

from xls.common import runfiles
from xls.ir import op_pb2
from xls.ir.python import package
from xls.solvers.python import lec_characterizer
from absl.testing import absltest


class LecCharacterizerTest(absltest.TestCase):

  def test_generates_sources(self):
    server_path = runfiles.get_path('xls/synthesis/dummy_synthesis_server_main')

    port = portpicker.pick_unused_port()
    lc = lec_characterizer.LecCharacterizer(
        [server_path, '--port={}'.format(port)], port)
    p = package.Package('the_package')
    ir_text, netlist_text = lc._generate_sources(
        op_pb2.OpProto.OP_ADD,
        [p.get_bits_type(8), p.get_bits_type(8)], p.get_bits_type(8))
    self.assertIn('ret add.1: bits[8]', ir_text)
    self.assertEqual(netlist_text, '// NETLIST')


if __name__ == '__main__':
  absltest.main()

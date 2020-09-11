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

import os
import tempfile
import portpicker

from google.protobuf import text_format
from absl.testing import absltest
from xls.common import gfile
from xls.common import runfiles
from xls.ir import op_pb2
from xls.ir import xls_type_pb2
from xls.ir.python import package
from xls.solvers.python import lec_characterizer
from xls.solvers.python import lec_characterizer_pb2


class LecCharacterizerTest(absltest.TestCase):

  _CELL_LIBRARY_PATH = 'xls/netlist/fake_cell_library.textproto'

  def setUp(self):
    super().setUp()
    server_path = runfiles.get_path('xls/synthesis/dummy_synthesis_server_main')
    self.port = portpicker.pick_unused_port()
    self.lc = lec_characterizer.LecCharacterizer(
        [server_path, '--port={}'.format(self.port)], self.port)

    cell_lib_path = runfiles.get_path(self._CELL_LIBRARY_PATH)
    with gfile.open(cell_lib_path, 'r') as f:
      self.cell_lib_text = f.read()

  def tearDown(self):
    super().tearDown()
    portpicker.return_port(self.port)

  # Smoke test showing we're able to generate IR/netlist sources.
  def test_generates_sources(self):
    p = package.Package('the_package')
    ir_text, netlist_text = self.lc._generate_sources(
        op_pb2.OpProto.OP_ADD,
        [p.get_bits_type(8), p.get_bits_type(8)], p.get_bits_type(8))
    self.assertIn('ret add.1: bits[8]', ir_text)
    self.assertEqual(netlist_text, '// NETLIST')

  # Tests that an extremely simple case runs without exploding.
  def test_lec_smoke(self):
    p = package.Package('the_package')

    temp_dir = tempfile.TemporaryDirectory()
    results_path = os.path.join(temp_dir.name, 'results.textproto')

    num_iters = 16

    byte_type = p.get_bits_type(8)
    self.lc.run(
        op=op_pb2.OpProto.OP_ADD,
        samples=[([byte_type, byte_type], byte_type)],
        num_iters=num_iters,
        cell_library_textproto=self.cell_lib_text,
        results_path=results_path,
        lec_fn=lambda a, b, c, d: True)

    # Open results, verify contents
    results = lec_characterizer_pb2.LecTiming()
    with gfile.open(results_path, 'r') as f:
      text_format.Parse(f.read(), results)

    self.assertEqual(results.ir_function, 'single_op_OP_ADD')
    self.assertLen(results.test_cases, 1)
    test_case = results.test_cases[0]
    self.assertLen(test_case.exec_times_us, num_iters)

  # Tests that we can correctly append to a preexisting proto file.
  def test_read_then_write(self):
    p = package.Package('the_package')

    temp_dir = tempfile.TemporaryDirectory()
    results_path = os.path.join(temp_dir.name, 'results.textproto')
    results = lec_characterizer_pb2.LecTiming()
    results.ir_function = 'single_op_OP_ADD'

    # Add one un-touched test case, and add one that should be appended to.
    proto_byte = xls_type_pb2.TypeProto()
    proto_byte.type_enum = xls_type_pb2.TypeProto.BITS
    proto_byte.bit_count = 8
    proto_short = xls_type_pb2.TypeProto()
    proto_short.type_enum = xls_type_pb2.TypeProto.BITS
    proto_short.bit_count = 16

    test_case = results.test_cases.add()
    param = test_case.function_type.parameters.add()
    param.CopyFrom(proto_short)
    param = test_case.function_type.parameters.add()
    param.CopyFrom(proto_short)
    test_case.function_type.return_type.CopyFrom(proto_short)

    test_case = results.test_cases.add()
    param = test_case.function_type.parameters.add()
    param.CopyFrom(proto_byte)
    param = test_case.function_type.parameters.add()
    param.CopyFrom(proto_byte)
    test_case.function_type.return_type.CopyFrom(proto_byte)
    test_case.exec_times_us.extend([1, 3, 7])
    test_case.average_us = 3

    with gfile.open(results_path, 'w') as f:
      f.write(text_format.MessageToString(results))

    num_iters = 16
    byte_type = p.get_bits_type(8)
    self.lc.run(
        op=op_pb2.OpProto.OP_ADD,
        samples=[([byte_type, byte_type], byte_type)],
        num_iters=num_iters,
        cell_library_textproto=self.cell_lib_text,
        results_path=results_path,
        lec_fn=lambda a, b, c, d: True)

    results = lec_characterizer_pb2.LecTiming()
    with gfile.open(results_path, 'r') as f:
      text_format.Parse(f.read(), results)

    self.assertEqual(results.ir_function, 'single_op_OP_ADD')
    self.assertLen(results.test_cases, 2)
    for test_case in results.test_cases:
      if test_case.function_type.return_type.bit_count == 16:
        self.assertEmpty(test_case.exec_times_us)
      else:
        self.assertLen(test_case.exec_times_us, 3 + num_iters)


if __name__ == '__main__':
  absltest.main()

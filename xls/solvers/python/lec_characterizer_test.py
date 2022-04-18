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
"""Tests for xls.solvers.lec_characterizer."""

import subprocess

import portpicker

from absl.testing import absltest
from xls.common import gfile
from xls.common import runfiles
from xls.ir import op_pb2
from xls.ir import xls_type_pb2
from xls.solvers.python import lec_characterizer
from xls.solvers.python import lec_characterizer_pb2


class LecCharacterizerTest(absltest.TestCase):

  _CELL_LIBRARY_PATH = 'xls/netlist/fake_cell_library.textproto'

  def setUp(self):
    super().setUp()
    server_path = runfiles.get_path('xls/synthesis/dummy_synthesis_server_main')
    self._port = portpicker.pick_unused_port()
    self._synthesis_server = subprocess.Popen(
        [server_path, '--port={}'.format(self._port)], self._port)

    cell_lib_path = runfiles.get_path(self._CELL_LIBRARY_PATH)
    with gfile.open(cell_lib_path, 'r') as f:
      self._cell_lib_text = f.read()

    self._lc = lec_characterizer.LecCharacterizer('localhost:{}'.format(
        self._port))

    self._byte_type = xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.BITS, bit_count=8)

  def tearDown(self):
    super().tearDown()
    if self._synthesis_server:
      self._synthesis_server.kill()
      self._synthesis_server.wait()
    portpicker.return_port(self._port)

  # Smoke test showing we're able to generate IR/netlist sources.
  def test_generates_sources(self):
    ir_text, netlist_text = self._lc._generate_sources(
        op_pb2.OpProto.OP_ADD, [self._byte_type, self._byte_type],
        self._byte_type)
    self.assertIn('ret result: bits[8]', ir_text)
    self.assertEqual(netlist_text, '// NETLIST')

  # Tests that an extremely simple case runs without exploding.
  def test_lec_smoke(self):
    num_iters = 16
    results = lec_characterizer_pb2.LecTiming()
    function_type = xls_type_pb2.FunctionTypeProto()
    function_type.parameters.add().CopyFrom(self._byte_type)
    function_type.parameters.add().CopyFrom(self._byte_type)
    function_type.return_type.CopyFrom(self._byte_type)

    self._lc.run(
        results=results,
        op=op_pb2.OpProto.OP_ADD,
        function_type=function_type,
        num_iters=num_iters,
        cell_library_textproto=self._cell_lib_text,
        lec_fn=lambda a, b, c, d: True,
        results_fn=lambda x: None)

    self.assertLen(results.test_cases, 1)
    test_case = results.test_cases[0]
    self.assertLen(test_case.exec_times_us, num_iters)

  # Tests that we can correctly append to a preexisting proto.
  def test_appends_to_existing(self):
    results = lec_characterizer_pb2.LecTiming()
    results.ir_function = 'single_op_OP_ADD'

    # Add one un-touched test case, and add one that should be appended to.
    proto_short = xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.BITS, bit_count=16)

    test_case = results.test_cases.add()
    test_case.function_type.parameters.add().CopyFrom(proto_short)
    test_case.function_type.parameters.add().CopyFrom(proto_short)
    test_case.function_type.return_type.CopyFrom(proto_short)
    test_case.exec_times_us.extend([1, 3, 7])
    test_case.average_us = 3

    num_iters = 16
    self._lc.run(
        results=results,
        op=op_pb2.OpProto.OP_ADD,
        function_type=test_case.function_type,
        num_iters=num_iters,
        cell_library_textproto=self._cell_lib_text,
        lec_fn=lambda a, b, c, d: True,
        results_fn=lambda x: None)

    self.assertLen(test_case.exec_times_us, 3 + num_iters)

  # Tests that we can correctly handle array types in the flow.
  def test_handles_arrays(self):
    num_iters = 16
    results = lec_characterizer_pb2.LecTiming()
    byte_array_type = xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.TypeEnum.ARRAY, array_size=8)
    byte_array_type.array_element.CopyFrom(self._byte_type)

    function_type = xls_type_pb2.FunctionTypeProto()
    function_type.parameters.add().CopyFrom(byte_array_type)
    function_type.parameters.add().CopyFrom(self._byte_type)
    function_type.parameters.add().CopyFrom(self._byte_type)
    function_type.return_type.CopyFrom(byte_array_type)

    self._lc.run(
        results=results,
        op=op_pb2.OpProto.OP_ARRAY_UPDATE,
        function_type=function_type,
        num_iters=num_iters,
        cell_library_textproto=self._cell_lib_text,
        lec_fn=lambda a, b, c, d: True,
        results_fn=lambda x: None)

    self.assertLen(results.test_cases, 1)
    test_case = results.test_cases[0]
    self.assertLen(test_case.exec_times_us, num_iters)

  # Tests that we can correctly handle tuple types in the flow.
  def test_handles_tuples(self):
    num_iters = 16
    results = lec_characterizer_pb2.LecTiming()
    tuple_type = xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.TypeEnum.TUPLE)
    tuple_type.tuple_elements.add().CopyFrom(self._byte_type)
    tuple_type.tuple_elements.add().CopyFrom(self._byte_type)
    tuple_type.tuple_elements.add().CopyFrom(self._byte_type)

    function_type = xls_type_pb2.FunctionTypeProto()
    function_type.parameters.add().CopyFrom(self._byte_type)
    function_type.parameters.add().CopyFrom(self._byte_type)
    function_type.parameters.add().CopyFrom(self._byte_type)
    function_type.return_type.CopyFrom(tuple_type)

    self._lc.run(
        results=results,
        op=op_pb2.OpProto.OP_TUPLE,
        function_type=function_type,
        num_iters=num_iters,
        cell_library_textproto=self._cell_lib_text,
        lec_fn=lambda a, b, c, d: True,
        results_fn=lambda x: None)

    self.assertLen(results.test_cases, 1)
    test_case = results.test_cases[0]
    self.assertLen(test_case.exec_times_us, num_iters)


if __name__ == '__main__':
  absltest.main()

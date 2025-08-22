# Copyright 2025 The XLS Authors
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

"""Tests for xls.tools.block_to_verilog_main."""

import subprocess
from google.protobuf import text_format

from absl.testing import absltest
from xls.common import test_base
from xls.common import runfiles
from xls.codegen import module_signature_pb2


BLOCK_TO_VERILOG_MAIN_PATH = runfiles.get_path('xls/tools/block_to_verilog_main')

COMBINATIONAL_BLOCK_IR = '''package add

#[signature("""module_name: "my_function" data_ports { direction: PORT_DIRECTION_INPUT name: "a" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_INPUT name: "b" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_OUTPUT name: "out" width: 32 type { type_enum: BITS bit_count: 32 } } 
                                          fixed_latency { latency: 0 } """)]

top block my_function(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=6)
  b: bits[32] = input_port(name=b, id=7)
  sum: bits[32] = add(a, b, id=8)
  not_sum: bits[32] = not(sum, id=9)
  not_not_sum: bits[32] = not(not_sum, id=10)
  out: () = output_port(not_not_sum, name=out, id=11)
}
'''

INSTANTIATION_IR = '''package inst

block sub_block(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in, id=1)
  out: () = output_port(in, name=out, id=2)
}

// Signature for the top block module interface.
#[signature("""module_name: "my_function" data_ports { direction: PORT_DIRECTION_INPUT name: "a" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_OUTPUT name: "out" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          fixed_latency { latency: 0 } """)]
top block my_function(a: bits[32], out: bits[32]) {
  instantiation my_inst(block=sub_block, kind=block)
  a: bits[32] = input_port(name=a, id=3)
  a_in: () = instantiation_input(a, instantiation=my_inst, port_name=in, id=4)
  y: bits[32] = instantiation_output(instantiation=my_inst, port_name=out, id=5)
  out: () = output_port(y, name=out, id=6)
}
'''

class BlockToVerilogMainTest(absltest.TestCase):

  def test_block_ir_generates_verilog(self):
    # Use a simple combinational block IR directly.
    block_ir_file = self.create_tempfile(content=COMBINATIONAL_BLOCK_IR)
    verilog_path = test_base.create_named_output_text_file('my_function.v')
    signature_path = test_base.create_named_output_text_file(
        'my_function.sig.textproto'
    )
    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--output_signature_path=' + signature_path,
        '--output_verilog_path=' + verilog_path,
        block_ir_file.full_path,
    ])

    with open(verilog_path, 'r') as f:
      verilog = f.read()
      # Basic sanity checks on generated Verilog.
      self.assertIn('module my_function(', verilog)
      self.assertIn('endmodule', verilog)
      # Expect at least one input and output port declaration.
      self.assertIn('input wire', verilog)
      self.assertIn('output wire', verilog)

    # Parse and sanity-check the emitted signature textproto.
    with open(signature_path, 'r') as f:
      sig_proto = text_format.Parse(
          f.read(), module_signature_pb2.ModuleSignatureProto()
      )
      self.assertEqual(sig_proto.module_name, 'my_function')
      # Expect 3 data ports: a (in), b (in), out (out).
      self.assertLen(sig_proto.data_ports, 3)
      names = [p.name for p in sig_proto.data_ports]
      self.assertEqual(names, ['a', 'b', 'out'])
      dirs = [p.direction for p in sig_proto.data_ports]
      # 1=input, 2=output
      self.assertEqual(dirs, [1, 1, 2])

  def test_block_ir_with_instantiation(self):
    ir_file = self.create_tempfile(content=INSTANTIATION_IR)
    verilog_path = test_base.create_named_output_text_file('inst_block.v')

    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--output_verilog_path=' + verilog_path,
        ir_file.full_path,
    ])

    with open(verilog_path, 'r') as f:
      verilog = f.read()
      self.assertIn('module my_function(', verilog)
      self.assertIn('module sub_block', verilog)
      self.assertIn('sub_block my_inst', verilog)


if __name__ == '__main__':
  absltest.main()

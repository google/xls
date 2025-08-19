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

import subprocess
from google.protobuf import text_format
from absl.testing import absltest
from xls.codegen import codegen_residual_data_pb2
from xls.codegen import module_signature_pb2
from xls.common import runfiles
from xls.common import test_base


BLOCK_TO_VERILOG_MAIN_PATH = runfiles.get_path(
    'xls/tools/block_to_verilog_main'
)

BLOCK_IR = '''package add

#[signature("""module_name: "my_function" data_ports { direction: PORT_DIRECTION_INPUT name: "a" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_INPUT name: "b" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_OUTPUT name: "out" width: 32 type { type_enum: BITS bit_count: 32 } } 
                                          fixed_latency { latency: 0 } """)]

top block my_function(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=6)
  b: bits[32] = input_port(name=b, id=7)
  neg_a: bits[32] = neg(a, id=8)
  not_b: bits[32] = not(b, id=9)
  id_not_b: bits[32] = identity(not_b, id=10)
  sum: bits[32] = add(neg_a, id_not_b, id=11)
  out: () = output_port(sum, name=out, id=12)
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

  def write_residual_data(self, node_pairs, filename):
    ref = codegen_residual_data_pb2.CodegenResidualData()
    b = ref.blocks.add()
    b.block_name = 'my_function'
    for name, node_id in node_pairs:
      n = b.nodes.add()
      n.node_name = name
      n.node_id = node_id
    path = test_base.create_named_output_text_file(filename)
    with open(path, 'w') as f:
      f.write(text_format.MessageToString(ref))
    return path

  def find_line_number(self, text: str, substring: str) -> int:
    """Returns line number of first occurrence of substring, or fails if not found."""
    for i, line in enumerate(text.splitlines(), start=1):
      if substring in line:
        return i
    self.fail(f"Substring '{substring}' not found in provided text")

  def test_block_ir_generates_verilog(self):
    # Use a simple combinational block IR directly.
    block_ir_file = self.create_tempfile(content=BLOCK_IR)
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
      self.assertIn('module my_function(', verilog)
      self.assertIn('endmodule', verilog)
      self.assertIn('input wire', verilog)
      self.assertIn('output wire', verilog)
    with open(signature_path, 'r') as f:
      sig_proto = text_format.Parse(
          f.read(), module_signature_pb2.ModuleSignatureProto()
      )
      self.assertEqual(sig_proto.module_name, 'my_function')
      self.assertLen(sig_proto.data_ports, 3)
      names = [p.name for p in sig_proto.data_ports]
      self.assertEqual(names, ['a', 'b', 'out'])
      dirs = [p.direction for p in sig_proto.data_ports]
      # 1=input, 2=output
      self.assertEqual(dirs, [1, 1, 2])

  def test_block_ir_with_instantiation(self):
    ir_file = self.create_tempfile(content=INSTANTIATION_IR)
    verilog_path = test_base.create_named_output_text_file('inst_block.v')
    residual_path = test_base.create_named_output_text_file('inst_residual.textproto')

    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--output_residual_data_path=' + residual_path,
        '--output_verilog_path=' + verilog_path,
        ir_file.full_path,
    ])
    with open(verilog_path, 'r') as f:
      verilog = f.read()
      self.assertIn('module my_function(', verilog)
      self.assertIn('module sub_block', verilog)
      self.assertIn('sub_block my_inst', verilog)

    # Verify residual data contains entries for both sub_block and top block.
    with open(residual_path, 'r') as f:
      residual = text_format.Parse(
          f.read(), codegen_residual_data_pb2.CodegenResidualData()
      )
    block_names = {b.block_name for b in residual.blocks}
    self.assertIn('sub_block', block_names)
    self.assertIn('my_function', block_names)

  def test_block_ir_with_specified_order(self):
    block_ir_file = self.create_tempfile(content=BLOCK_IR)

    def write_residual(node_pairs, filename):
      ref = codegen_residual_data_pb2.CodegenResidualData()
      b = ref.blocks.add()
      b.block_name = 'my_function'
      for name, node_id in node_pairs:
        n = b.nodes.add()
        n.node_name = name
        n.node_id = node_id
      path = test_base.create_named_output_text_file(filename)
      with open(path, 'w') as f:
        f.write(text_format.MessageToString(ref))
      return path

    # Provide name/id pairs corresponding to BLOCK_IR ids.
    ref1_path = self.write_residual_data([
        ('neg_a', 8), ('not_b', 9), ('id_not_b', 10),
    ], 'ref_block_order1.textproto')
    verilog1_path = test_base.create_named_output_text_file('ordered1.v')

    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--separate_lines',
        '--generator=combinational',
        '--output_verilog_path=' + verilog1_path,
        '--reference_residual_data_path=' + ref1_path,
        block_ir_file.full_path,
    ])
    with open(verilog1_path, 'r') as f:
      v1 = f.read()
      # Sanity-check the module is emitted and matches expected top name.
      self.assertIn('module my_function(', v1)
      self.assertLess(
          self.find_line_number(v1, 'assign neg_a ='),
          self.find_line_number(v1, 'assign not_b ='),
      )
      self.assertLess(
          self.find_line_number(v1, 'assign not_b ='),
          self.find_line_number(v1, 'assign id_not_b ='),
      )

    ref2_path = self.write_residual_data([
        ('not_b', 9), ('neg_a', 8), ('id_not_b', 10),
    ], 'ref_block_order2.textproto') 
    verilog2_path = test_base.create_named_output_text_file('ordered2.v')

    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--separate_lines',
        '--generator=combinational',
        '--output_verilog_path=' + verilog2_path,
        '--reference_residual_data_path=' + ref2_path,
        block_ir_file.full_path,
    ])
    with open(verilog2_path, 'r') as f:
      v2 = f.read()
      self.assertIn('module my_function(', v2)
      self.assertLess(
          self.find_line_number(v2, 'assign not_b ='),
          self.find_line_number(v2, 'assign neg_a ='),
      )
      self.assertLess(
          self.find_line_number(v2, 'assign neg_a ='),
          self.find_line_number(v2, 'assign id_not_b ='),
      )

  def test_block_ir_residual_roundtrip(self):
    # Emit residual in first invocation, consume as reference in second.
    block_ir_file = self.create_tempfile(content=BLOCK_IR)

    residual_path = test_base.create_named_output_text_file('residual_roundtrip.textproto')
    verilog1_path = test_base.create_named_output_text_file('roundtrip1.v')

    # First run: write Verilog and residual data.
    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--generator=combinational',
        '--output_verilog_path=' + verilog1_path,
        '--output_residual_data_path=' + residual_path,
        block_ir_file.full_path,
    ])

    # Verify residual contents and node count.
    with open(residual_path, 'r') as f:
      residual = text_format.Parse(
          f.read(), codegen_residual_data_pb2.CodegenResidualData()
      )
      self.assertEqual(len(residual.blocks), 1)

    # Second run: consume the residual as reference and ensure identical output.
    verilog2_path = test_base.create_named_output_text_file('roundtrip2.v')
    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--generator=combinational',
        '--output_verilog_path=' + verilog2_path,
        '--reference_residual_data_path=' + residual_path,
        block_ir_file.full_path,
    ])

    with open(verilog1_path, 'r') as f1, open(verilog2_path, 'r') as f2:
      self.assertEqual(f1.read(), f2.read())

  def test_pipeline_generator_rejects_reference_residual(self):
    # Using reference residual data with pipeline generator should fail.
    block_ir_file = self.create_tempfile(content=BLOCK_IR)

    ref_path = self.write_residual_data([
        ('neg_a', 8), ('not_b', 9), ('id_not_b', 10),
    ], 'ref_pipeline_order.textproto')

    proc = subprocess.run([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--generator=pipeline',
        '--reference_residual_data_path=' + ref_path,
        block_ir_file.full_path,
    ], capture_output=True, text=True)

    self.assertNotEqual(proc.returncode, 0)
    self.assertIn('not supported when generating pipelines', proc.stderr)


if __name__ == '__main__':
  absltest.main()
